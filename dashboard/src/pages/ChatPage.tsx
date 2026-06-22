import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { Alert, Button, Input, Tag, Tooltip, Typography, message as antdMessage } from 'antd'
import {
  LoadingOutlined,
  RobotOutlined,
  SendOutlined,
  UserOutlined,
} from '@ant-design/icons'
import type {
  ExecutionPlan,
  IntentAnalysis,
  InvokeResponse,
  OpsExecutionStepRecord,
  OpsSkillCallRecord,
  PlanProgressEntry,
  PlanStep,
} from '../types'
import { apiClient } from '../api/client'

type ChatMessageRole = 'user' | 'assistant' | 'system'

interface ChatMessage {
  id: string
  role: ChatMessageRole
  content: string
  createdAt: number
  meta?: InvokeResponse
  approvalResolved?: boolean
  pending?: boolean
  // 仅在 baselineTurn > 0 时为审批后追加的"延续消息"：
  //   - 仅渲染 turn > baselineTurn 的执行轮次（避免重复显示历史轮次）
  //   - 不再渲染意图分析 / 规划计划卡片（这两块已经在首条 assistant 消息里展示过）
  baselineTurn?: number
}

let _msgSeq = 0
function nextId(prefix: string) {
  _msgSeq += 1
  return `${prefix}-${Date.now()}-${_msgSeq}`
}

function formatTime(ts: number) {
  const d = new Date(ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function formatK(n: unknown): string {
  const v = typeof n === 'number' ? n : Number(n)
  if (!Number.isFinite(v) || v <= 0) return '0'
  if (v < 1000) return String(v)
  return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`
}

function tokenUsageColor(percent: unknown): string {
  const v = typeof percent === 'number' ? percent : Number(percent)
  if (!Number.isFinite(v)) return 'green'
  if (v >= 80) return 'red'
  if (v >= 50) return 'orange'
  return 'green'
}

function toRouteLabel(route?: string | null) {
  switch (route) {
    case 'dialog':
      return '通用对话'
    case 'ops':
      return '运维排障'
    case 'rag':
      return '知识库检索'
    default:
      return route ?? '未知'
  }
}

function toWorkflowLabel(status?: string | null) {
  switch (status) {
    case 'running':
      return '处理中'
    case 'waiting_approval':
      return '等待审批'
    case 'paused':
      return '已暂停'
    case 'completed':
      return '已完成'
    case 'rejected':
      return '已拒绝'
    default:
      return status ?? '-'
  }
}

function extractApprovalInterrupt(meta?: InvokeResponse) {
  if (!meta) return undefined
  const interrupt = meta.pending_interrupts?.find((item) => {
    const value = item.value as Record<string, unknown> | undefined
    return value?.type === 'approval_required'
  })
  if (!interrupt) return undefined

  const value = interrupt.value as Record<string, unknown> | undefined
  const payload =
    value?.payload && typeof value.payload === 'object'
      ? (value.payload as Record<string, unknown>)
      : undefined
  const payloadType = String(payload?.type ?? '')

  // 新格式：skill_invocation —— 由 skill_router 产生，单 skill 单次
  if (payloadType === 'skill_invocation') {
    const skillId = String(payload?.skill_id ?? '')
    const args = (payload?.arguments as Record<string, unknown>) ?? {}
    const risk = String(payload?.risk ?? 'medium').toLowerCase()
    const argsText = Object.keys(args).length > 0 ? JSON.stringify(args, null, 2) : ''
    const previewCmd =
      typeof args.cmd === 'string' && args.cmd.trim().length > 0
        ? String(args.cmd)
        : argsText || '(no arguments)'
    return {
      interruptId: String(interrupt.interrupt_id ?? ''),
      payload,
      actions: [] as Array<Record<string, unknown>>,
      commands: [{ cmd: previewCmd, title: skillId, risk }],
      autoExecutedLowRisk: [] as Array<{ cmd: string; title: string }>,
      reason: String(payload?.reason ?? `即将调用 skill: ${skillId}`),
      thought: String(payload?.thought ?? ''),
      skillId,
    }
  }

  // 旧格式：actions / commands / auto_executed_low_risk
  const actions = Array.isArray(payload?.actions)
    ? (payload.actions as Array<Record<string, unknown>>)
    : []
  const commands = Array.isArray(payload?.commands)
    ? (payload.commands as Array<unknown>).map((c) => {
        if (typeof c === 'string') {
          return { cmd: c, title: '', risk: 'medium' }
        }
        if (c && typeof c === 'object') {
          const obj = c as Record<string, unknown>
          return {
            cmd: String(obj.cmd ?? ''),
            title: String(obj.title ?? ''),
            risk: String(obj.risk ?? 'medium').toLowerCase(),
          }
        }
        return { cmd: JSON.stringify(c), title: '', risk: 'medium' }
      })
    : []
  const autoExecutedLowRisk = Array.isArray(payload?.auto_executed_low_risk)
    ? (payload.auto_executed_low_risk as Array<Record<string, unknown>>).map((item) => ({
        cmd: String(item.cmd ?? ''),
        title: String(item.title ?? ''),
      }))
    : []

  return {
    interruptId: String(interrupt.interrupt_id ?? ''),
    payload,
    actions,
    commands,
    autoExecutedLowRisk,
    reason: String(payload?.reason ?? '需要你的确认'),
    thought: '',
    skillId: '',
  }
}

function riskColor(risk?: string): string {
  const r = String(risk ?? 'medium').toLowerCase()
  if (r === 'high') return 'red'
  if (r === 'low') return 'green'
  return 'orange'
}

interface RoundBubbleData {
  turn: number
  analysis: string[]
  tools: Array<{ label: string; status: string; durationMs?: number }>
  commands: Array<{ title: string; cmd: string; risk?: string }>
  excerpts: string[]
  status: string
}

function normalizePlainText(input: string): string {
  return input
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^\s{0,3}#{1,6}\s*/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\|.*\|\s*$/gm, '')
    .replace(/^\s*:?-{3,}:?\s*$/gm, '')
    .replace(/^\s*\d+\.\s*(现状摘要|最可能根因.*|建议清单|建议命令)\s*$/gm, '')
    .replace(/^\s*(现状摘要|最可能根因.*|建议清单|建议命令)\s*$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function toReasoningLine(input: string): string {
  const text = normalizePlainText(input)
    .replace(/^分析[:：]?\s*/i, '')
    .replace(/^思路[:：]?\s*/i, '')
    .replace(/^判断[:：]?\s*/i, '')
    .replace(/^我(?:先|需要|准备|会)?/g, '')
    .trim()
  if (!text) return ''

  const firstLine = text.split(/\n+/)[0]?.trim() ?? ''
  const firstSentence =
    firstLine.split(/(?<=[。！？!?;；])/)[0]?.trim() || firstLine

  return firstSentence.length > 60
    ? `${firstSentence.slice(0, 60).trim()}...`
    : firstSentence
}

function pushUnique(list: string[], value: string, limit = 3) {
  const text = normalizePlainText(value)
  if (!text || list.includes(text) || list.length >= limit) return
  list.push(text)
}

function bubbleStatusClass(status: string): string {
  switch (status) {
    case 'failed':
      return 'chat-mini-bubble-error'
    case 'pending_approval':
    case 'planned':
      return 'chat-mini-bubble-warn'
    default:
      return 'chat-mini-bubble-neutral'
  }
}

function complexityColor(c?: string): string {
  switch (String(c || '').toLowerCase()) {
    case 'simple':
      return 'green'
    case 'complex':
      return 'red'
    default:
      return 'gold'
  }
}

function planStepStatusColor(s?: string): string {
  switch (String(s || '').toLowerCase()) {
    case 'done':
      return 'green'
    case 'failed':
      return 'red'
    case 'running':
      return 'blue'
    case 'pending_approval':
      return 'orange'
    case 'skipped':
      return 'default'
    default:
      return 'cyan' // planned
  }
}

function planStepStatusLabel(s?: string): string {
  switch (String(s || '').toLowerCase()) {
    case 'done':
      return '已完成'
    case 'failed':
      return '失败'
    case 'running':
      return '执行中'
    case 'pending_approval':
      return '待审批'
    case 'skipped':
      return '已跳过'
    case 'planned':
      return '已规划'
    default:
      return s || '已规划'
  }
}

interface IntentCardProps {
  intent: IntentAnalysis
}

function IntentCard({ intent }: IntentCardProps) {
  const summary = String(intent.summary ?? '').trim()
  const subs = (intent.sub_questions ?? []).filter(Boolean)
  const hints = (intent.domain_hints ?? []).filter(Boolean)
  const reasoning = String(intent.reasoning ?? '').trim()
  if (!summary && subs.length === 0 && hints.length === 0) return null

  return (
    <div className="chat-mini-bubble chat-mini-bubble-neutral">
      <div className="chat-mini-title">
        <span>意图分析</span>
        {intent.complexity ? (
          <Tag color={complexityColor(intent.complexity)} style={{ marginInlineStart: 8 }}>
            {String(intent.complexity)}
          </Tag>
        ) : null}
        {typeof intent.need_tools === 'boolean' ? (
          <Tag
            color={intent.need_tools ? 'geekblue' : 'default'}
            style={{ marginInlineStart: 4 }}
          >
            {intent.need_tools ? '需要工具' : '无需工具'}
          </Tag>
        ) : null}
      </div>
      {summary ? (
        <div style={{ fontSize: 13, color: '#222', marginBottom: 6 }}>{summary}</div>
      ) : null}
      {subs.length > 0 ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
          {subs.map((q, i) => (
            <Tag key={`sub-${i}`} color="blue" style={{ whiteSpace: 'normal' }}>
              {q}
            </Tag>
          ))}
        </div>
      ) : null}
      {hints.length > 0 ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
          {hints.map((h, i) => (
            <Tag key={`hint-${i}`}>{h}</Tag>
          ))}
        </div>
      ) : null}
      {reasoning ? (
        <Typography.Paragraph
          type="secondary"
          style={{ fontSize: 12, marginBottom: 0 }}
          ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
        >
          {reasoning}
        </Typography.Paragraph>
      ) : null}
    </div>
  )
}

interface PlanCardProps {
  plan: ExecutionPlan
  progress?: PlanProgressEntry[]
}

function PlanCard({ plan, progress }: PlanCardProps) {
  const steps: PlanStep[] = (plan.steps ?? []).filter(Boolean)
  const reasoning = String(plan.reasoning ?? '').trim()
  if (steps.length === 0) {
    if (!reasoning) return null
    return (
      <div className="chat-mini-bubble chat-mini-bubble-neutral">
        <div className="chat-mini-title">规划计划</div>
        <Typography.Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 0 }}>
          无需调用工具：{reasoning}
        </Typography.Paragraph>
      </div>
    )
  }

  const progressByStep = new Map<string, PlanProgressEntry>()
  for (const p of progress ?? []) {
    if (p.step_id) progressByStep.set(p.step_id, p)
  }

  return (
    <div className="chat-mini-bubble chat-mini-bubble-neutral">
      <div className="chat-mini-title">规划计划 · 共 {steps.length} 步</div>
      {reasoning ? (
        <Typography.Paragraph
          type="secondary"
          style={{ fontSize: 12, marginBottom: 8 }}
          ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
        >
          {reasoning}
        </Typography.Paragraph>
      ) : null}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {steps.map((step, idx) => {
          const matched = progressByStep.get(step.id)
          const status = matched?.status || step.status || 'planned'
          return (
            <div
              key={step.id || `step-${idx}`}
              style={{
                background: '#fafafa',
                borderRadius: 6,
                padding: '6px 10px',
                border: '1px solid #efefef',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ color: '#888', fontSize: 12 }}>步骤 {idx + 1}</span>
                <strong style={{ fontSize: 13 }}>{step.title || '(未命名)'}</strong>
                <Tag color={planStepStatusColor(status)}>{planStepStatusLabel(status)}</Tag>
                {step.suggested_skill_id ? (
                  <Tag color="purple">{step.suggested_skill_id}</Tag>
                ) : null}
              </div>
              {step.intent ? (
                <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{step.intent}</div>
              ) : null}
              {step.expected_output ? (
                <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
                  预期输出：{step.expected_output}
                </div>
              ) : null}
              {matched?.observation ? (
                <Typography.Paragraph
                  type="secondary"
                  style={{ fontSize: 12, marginTop: 4, marginBottom: 0 }}
                  ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
                >
                  观测：{matched.observation}
                </Typography.Paragraph>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function buildRounds(meta?: InvokeResponse): RoundBubbleData[] {
  if (!meta) return []

  const roundMap = new Map<number, RoundBubbleData>()
  const ensureRound = (turn: number) => {
    if (!roundMap.has(turn)) {
      roundMap.set(turn, {
        turn,
        analysis: [],
        tools: [],
        commands: [],
        excerpts: [],
        status: 'done',
      })
    }
    return roundMap.get(turn)!
  }

  const steps = (meta.ops_execution_steps ?? []) as OpsExecutionStepRecord[]
  const skillCalls = (meta.ops_skill_calls ?? []) as OpsSkillCallRecord[]
  const scratchpad = (meta.router_scratchpad ?? []) as Array<Record<string, unknown>>

  let fallbackTurn = 0
  const nextFallbackTurn = () => {
    fallbackTurn += 1
    return fallbackTurn
  }

  scratchpad.forEach((entry, index) => {
    const turn = Number(entry.turn ?? 0) || index + 1
    fallbackTurn = Math.max(fallbackTurn, turn)
    const round = ensureRound(turn)
    const skillId = String(entry.skill_id ?? '').trim()
    const action = String(entry.action ?? '').trim()
    const status = String(entry.status ?? (action || 'done')).trim()
    const thought = String(entry.thought ?? '').trim()
    const observation = String(entry.observation ?? '').trim()
    if (skillId) {
      round.tools.push({ label: skillId, status })
    } else if (action && action !== 'finalize') {
      round.tools.push({ label: action, status })
    }
    const reasoning = toReasoningLine(thought)
    if (reasoning) pushUnique(round.analysis, reasoning, 1)
    if (observation && action !== 'finalize') pushUnique(round.excerpts, observation, 2)
    if (status) round.status = status
  })

  skillCalls.forEach((call, index) => {
    const turn = Number(call.turn ?? 0) || nextFallbackTurn()
    const round = ensureRound(turn)
    const label = String(call.skill_id ?? call.name ?? `tool-${index + 1}`).trim()
    const status = String(call.status ?? call.result_status ?? round.status).trim() || 'done'
    const durationMs = Number(call.duration_ms ?? 0) || undefined
    if (label && !round.tools.some((item) => item.label === label && item.status === status)) {
      round.tools.push({ label, status, durationMs })
    }
    const cmd = String((call.arguments as Record<string, unknown> | undefined)?.cmd ?? '').trim()
    if (cmd && !round.commands.some((item) => item.cmd === cmd)) {
      round.commands.push({ title: label, cmd })
    }
    if (call.result_excerpt) pushUnique(round.excerpts, String(call.result_excerpt), 2)
    if (status) round.status = status
  })

  steps.forEach((step, index) => {
    const turn = Number(step.turn ?? 0) || nextFallbackTurn()
    const round = ensureRound(turn)
    const title = String(step.title ?? step.skill_id ?? `step-${index + 1}`).trim()
    const command = String(step.command ?? '').trim()
    const risk = String(step.risk ?? '').trim() || undefined
    const status = String(step.status ?? round.status).trim() || 'done'
    if (title && !round.tools.some((item) => item.label === title)) {
      round.tools.push({
        label: title,
        status,
        durationMs: Number(step.duration_ms ?? 0) || undefined,
      })
    }
    if (command && !round.commands.some((item) => item.cmd === command)) {
      round.commands.push({ title, cmd: command, risk })
    }
    if (step.result_excerpt) pushUnique(round.excerpts, String(step.result_excerpt), 2)
    if (status) round.status = status
  })

  return [...roundMap.values()].sort((a, b) => a.turn - b.turn)
}

/** 取 InvokeResponse 里执行轮次的最大 turn，作为下一条增量消息的基线。 */
function maxTurnFromMeta(meta?: InvokeResponse): number {
  if (!meta) return 0
  let max = 0
  const scan = (items: Array<Record<string, unknown>> | undefined) => {
    for (const item of items ?? []) {
      const t = Number(item?.turn ?? 0)
      if (Number.isFinite(t) && t > max) max = t
    }
  }
  scan(meta.router_scratchpad as Array<Record<string, unknown>> | undefined)
  scan(meta.ops_skill_calls as unknown as Array<Record<string, unknown>> | undefined)
  scan(meta.ops_execution_steps as unknown as Array<Record<string, unknown>> | undefined)
  return max
}

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [threadId, setThreadId] = useState<string>()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [approvalSubmitting, setApprovalSubmitting] = useState(false)
  const [error, setError] = useState<string>()
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages, loading, approvalSubmitting])

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === 'assistant' && m.meta),
    [messages],
  )

  const sendQuery = async () => {
    const query = input.trim()
    if (!query || loading) return

    const userMsg: ChatMessage = {
      id: nextId('user'),
      role: 'user',
      content: query,
      createdAt: Date.now(),
    }
    const pendingMsg: ChatMessage = {
      id: nextId('pending'),
      role: 'assistant',
      content: '正在思考...',
      createdAt: Date.now(),
      pending: true,
    }
    setMessages((prev) => [...prev, userMsg, pendingMsg])
    setInput('')
    setLoading(true)
    setError(undefined)

    try {
      const response = await apiClient.invoke({
        user_query: query,
        thread_id: threadId,
      })
      setThreadId(response.thread_id)
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== pendingMsg.id)
          .concat({
            id: nextId('assistant'),
            role: 'assistant',
            content: response.final_answer || '（无内容）',
            createdAt: Date.now(),
            meta: response,
          }),
      )
    } catch (err) {
      const msg = err instanceof Error ? err.message : '调用 /invoke 失败'
      setError(msg)
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== pendingMsg.id)
          .concat({
            id: nextId('error'),
            role: 'system',
            content: `请求失败：${msg}`,
            createdAt: Date.now(),
          }),
      )
    } finally {
      setLoading(false)
    }
  }

  const resolveApproval = async (messageId: string, approved: boolean) => {
    const target = messages.find((m) => m.id === messageId)
    if (!target?.meta?.thread_id) return
    setApprovalSubmitting(true)
    setError(undefined)

    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, approvalResolved: true } : m)),
    )
    setMessages((prev) => [
      ...prev,
      {
        id: nextId('sys'),
        role: 'system',
        content: approved ? '已批准本轮命令，继续排查...' : '已拒绝本轮命令，流程停止。',
        createdAt: Date.now(),
      },
    ])
    const pendingMsg: ChatMessage = {
      id: nextId('pending'),
      role: 'assistant',
      content: approved ? '正在执行下一轮命令...' : '正在结束流程...',
      createdAt: Date.now(),
      pending: true,
    }
    setMessages((prev) => [...prev, pendingMsg])

    try {
      const response = await apiClient.invoke({
        thread_id: target.meta.thread_id,
        resume: approved,
      })
      setThreadId(response.thread_id)
      // 审批后只展示新增轮次：用上一条 assistant 消息的最大 turn 作为基线
      const baselineTurn = maxTurnFromMeta(target.meta)
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== pendingMsg.id)
          .concat({
            id: nextId('assistant'),
            role: 'assistant',
            content: response.final_answer || '（无内容）',
            createdAt: Date.now(),
            meta: response,
            baselineTurn,
          }),
      )
      antdMessage.success(approved ? '已继续执行' : '已停止流程')
    } catch (err) {
      const msg = err instanceof Error ? err.message : '提交审批结果失败'
      setError(msg)
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== pendingMsg.id)
          .concat({
            id: nextId('error'),
            role: 'system',
            content: `审批提交失败：${msg}`,
            createdAt: Date.now(),
          }),
      )
    } finally {
      setApprovalSubmitting(false)
    }
  }

  const clearConversation = () => {
    setMessages([])
    setThreadId(undefined)
    setError(undefined)
    setInput('')
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    // 中文/日文/韩文等输入法正在组合时，回车用于确认候选词，不发送
    if (event.nativeEvent.isComposing || event.keyCode === 229) return
    event.preventDefault()
    if (loading || approvalSubmitting) return
    void sendQuery()
  }

  return (
    <div className="chat-root">
      <div className="chat-topbar">
        <div className="chat-topbar-left">
          <Typography.Text strong>Ops RAG Chat</Typography.Text>
          {threadId ? (
            <Tooltip title={threadId}>
              <Tag color="blue" style={{ marginInlineStart: 8 }}>
                thread · {threadId.slice(0, 8)}
              </Tag>
            </Tooltip>
          ) : (
            <Tag style={{ marginInlineStart: 8 }}>未创建会话</Tag>
          )}
          {lastAssistant?.meta?.route ? (
            <Tag color="geekblue">{toRouteLabel(lastAssistant.meta.route)}</Tag>
          ) : null}
          {lastAssistant?.meta?.workflow_status ? (
            <Tag color="purple">{toWorkflowLabel(lastAssistant.meta.workflow_status)}</Tag>
          ) : null}
          {lastAssistant?.meta?.token_usage ? (
            <Tooltip
              title={
                <div style={{ fontSize: 12, lineHeight: '18px' }}>
                  <div>输入: {Number(lastAssistant.meta.token_usage.last_prompt_tokens ?? 0)}</div>
                  <div>
                    回复: {Number(lastAssistant.meta.token_usage.last_completion_tokens ?? 0)}
                  </div>
                  <div>历史: {Number(lastAssistant.meta.token_usage.history_tokens ?? 0)}</div>
                  <div>摘要: {Number(lastAssistant.meta.token_usage.summary_tokens ?? 0)}</div>
                  <div>编码: {String(lastAssistant.meta.token_usage.encoding ?? '-')}</div>
                </div>
              }
            >
              <Tag color={tokenUsageColor(lastAssistant.meta.token_usage.percent)}>
                上下文 {formatK(lastAssistant.meta.token_usage.total)} /{' '}
                {formatK(lastAssistant.meta.token_usage.context_window)} ·{' '}
                {Number(lastAssistant.meta.token_usage.percent ?? 0).toFixed(1)}%
              </Tag>
            </Tooltip>
          ) : null}
        </div>
        <Button size="small" onClick={clearConversation}>
          清空会话
        </Button>
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat-empty">
            <RobotOutlined style={{ fontSize: 48, color: '#bbb' }} />
            <Typography.Paragraph type="secondary" style={{ marginTop: 16 }}>
              输入问题开始对话。支持通用对话、知识库检索、本机运维排障（需要逐轮审批）。
            </Typography.Paragraph>
            <Typography.Text type="secondary">Enter 发送，Shift+Enter 换行</Typography.Text>
          </div>
        ) : (
          messages.map((msg) => (
            <ChatBubble
              key={msg.id}
              message={msg}
              onApproval={resolveApproval}
              approvalSubmitting={approvalSubmitting}
            />
          ))
        )}
      </div>

      {error ? (
        <Alert
          type="error"
          showIcon
          message={error}
          closable
          onClose={() => setError(undefined)}
          style={{ margin: '0 16px' }}
        />
      ) : null}

      <div className="chat-inputbar">
        <Input.TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的问题... (Enter 发送，Shift+Enter 换行)"
          autoSize={{ minRows: 1, maxRows: 6 }}
          disabled={loading || approvalSubmitting}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          loading={loading}
          disabled={!input.trim() || approvalSubmitting}
          onClick={() => void sendQuery()}
        >
          发送
        </Button>
      </div>
    </div>
  )
}

interface ChatBubbleProps {
  message: ChatMessage
  approvalSubmitting: boolean
  onApproval: (messageId: string, approved: boolean) => void
}

function ChatBubble({ message, approvalSubmitting, onApproval }: ChatBubbleProps) {
  const isSystem = message.role === 'system'
  const isUser = message.role === 'user'

  // 计算模型输出（system/user 场景下仍然 hook，但值为空字符串，保持 Hooks 顺序稳定）
  const modelOutput = !isUser && !isSystem ? normalizePlainText(message.content) : ''

  // 模型输出"流式打字机"渲染（必须在任何 early return 之前，保持 Hooks 调用稳定）
  const [streamedOutput, setStreamedOutput] = useState('')
  useEffect(() => {
    if (!modelOutput || message.pending) {
      setStreamedOutput(modelOutput)
      return
    }
    setStreamedOutput('')
    const total = modelOutput.length
    const step = Math.max(1, Math.floor(total / 80))
    let cursor = 0
    let timer: ReturnType<typeof setInterval> | null = null
    timer = setInterval(() => {
      cursor = Math.min(total, cursor + step)
      setStreamedOutput(modelOutput.slice(0, cursor))
      if (cursor >= total && timer) {
        clearInterval(timer)
        timer = null
      }
    }, 24)
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [modelOutput, message.pending])

  if (isSystem) {
    return (
      <div className="chat-row chat-row-system">
        <div className="chat-system-pill">{message.content}</div>
      </div>
    )
  }

  const approval =
    !isUser && !message.approvalResolved ? extractApprovalInterrupt(message.meta) : undefined
  const recommendations = (!isUser && message.meta?.ops_recommendations) || []
  const citations = (!isUser && message.meta?.citations) || []
  const baselineTurn = message.baselineTurn ?? 0
  const isContinuation = baselineTurn > 0
  const allRounds = !isUser ? buildRounds(message.meta) : []
  // 审批后的延续消息只显示新增轮次
  const rounds = isContinuation
    ? allRounds.filter((r) => r.turn > baselineTurn)
    : allRounds
  const intentAnalysis = !isUser ? message.meta?.intent_analysis : undefined
  const executionPlan = !isUser ? message.meta?.execution_plan : undefined
  const planProgress = !isUser ? message.meta?.plan_progress : undefined
  // 延续消息不再重复渲染意图 / 规划卡片（已在第 1 条 assistant 消息里展示过）
  const showIntent =
    !isContinuation &&
    !!intentAnalysis &&
    (!!intentAnalysis.summary ||
      (intentAnalysis.sub_questions && intentAnalysis.sub_questions.length > 0) ||
      (intentAnalysis.domain_hints && intentAnalysis.domain_hints.length > 0))
  const showPlan =
    !isContinuation &&
    !!executionPlan &&
    ((executionPlan.steps && executionPlan.steps.length > 0) ||
      !!executionPlan.reasoning)

  const isStreaming = !isUser && !!modelOutput && streamedOutput.length < modelOutput.length

  return (
    <div className={`chat-row ${isUser ? 'chat-row-user' : 'chat-row-assistant'}`}>
      <div className="chat-avatar">{isUser ? <UserOutlined /> : <RobotOutlined />}</div>
      <div className="chat-bubble-wrap">
        <div className="chat-bubble-meta">
          <span>{isUser ? '你' : 'Ops RAG Agent'}</span>
          <span className="chat-bubble-time">{formatTime(message.createdAt)}</span>
        </div>
        {isUser || message.pending ? (
          <div
            className={`chat-bubble ${isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'} ${
              message.pending ? 'chat-bubble-pending' : ''
            }`}
          >
            {message.pending ? (
              <span>
                <LoadingOutlined /> {message.content}
              </span>
            ) : (
              <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
            )}
          </div>
        ) : null}

        {!isUser && showIntent && intentAnalysis ? (
          <IntentCard intent={intentAnalysis} />
        ) : null}

        {!isUser && showPlan && executionPlan ? (
          <PlanCard plan={executionPlan} progress={planProgress} />
        ) : null}

        {!isUser && rounds.length > 0 ? (
          <div className="chat-rounds">
            {rounds.map((round) => (
              <div key={`${message.id}-round-${round.turn}`} className="chat-round">
                <div className="chat-round-label">第 {round.turn} 轮</div>
                {round.analysis.map((item, index) => (
                  <div
                    key={`${message.id}-analysis-${round.turn}-${index}`}
                    className="chat-reasoning-line"
                  >
                    <span>{item}</span>
                  </div>
                ))}
                {round.tools.length > 0 ? (
                  <div className={`chat-mini-bubble ${bubbleStatusClass(round.status)}`}>
                    <div className="chat-mini-title">工具调用</div>
                    <div className="chat-tool-tags chat-tool-lines">
                      {round.tools.map((tool, index) => (
                        <div key={`${message.id}-tool-${round.turn}-${index}`} className="chat-tool-line">
                          <span className="chat-tool-call">已调用 {tool.label}</span>
                          {tool.durationMs ? (
                            <span className="chat-tool-duration">{tool.durationMs} ms</span>
                          ) : null}
                          <Tag color={tool.status === 'failed' ? 'red' : tool.status === 'planned' ? 'orange' : 'cyan'}>
                            {tool.status}
                          </Tag>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
                {round.commands.length > 0 ? (
                  <div className="chat-mini-bubble chat-mini-bubble-command">
                    <div className="chat-mini-title">命令</div>
                    {round.commands.map((item, index) => (
                      <div key={`${message.id}-cmd-${round.turn}-${index}`} className="chat-command-item">
                        <div className="chat-command-head">
                          <span>{item.title}</span>
                          {item.risk ? <Tag color={riskColor(item.risk)}>{item.risk}</Tag> : null}
                        </div>
                        <pre className="chat-step-cmd">$ {item.cmd}</pre>
                      </div>
                    ))}
                  </div>
                ) : null}
                {round.excerpts.map((excerpt, index) => (
                  <details
                    key={`${message.id}-excerpt-${round.turn}-${index}`}
                    className={`chat-mini-bubble ${bubbleStatusClass(round.status)} chat-excerpt-foldable`}
                  >
                    <summary className="chat-mini-title chat-excerpt-summary">
                      <span>运行结果</span>
                      <span className="chat-excerpt-hint">点击展开</span>
                    </summary>
                    <pre className="chat-step-excerpt">{excerpt.slice(0, 4000)}</pre>
                  </details>
                ))}
              </div>
            ))}
          </div>
        ) : null}

        {approval ? (
          <div className="chat-approval">
            <div className="chat-approval-head">
              <Tag color="orange">需要审批</Tag>
              <span className="chat-approval-reason">{approval.reason}</span>
            </div>
            {approval.thought ? (
              <div className="chat-approval-cmds">
                <div className="chat-approval-cmds-title">分析</div>
                <div style={{ color: '#555', fontSize: 12, whiteSpace: 'pre-wrap' }}>
                  {approval.thought}
                </div>
              </div>
            ) : null}
            {approval.autoExecutedLowRisk.length > 0 ? (
              <div className="chat-approval-cmds">
                <div className="chat-approval-cmds-title">本轮已自动执行（低风险）：</div>
                {approval.autoExecutedLowRisk.map((item, i) => (
                  <div key={`${message.id}-auto-${i}`} className="chat-approval-action">
                    <div>
                      <b>{item.title || item.cmd}</b>
                      <Tag color="green" style={{ marginInlineStart: 8 }}>
                        low
                      </Tag>
                    </div>
                    <pre className="chat-step-cmd">$ {item.cmd}</pre>
                  </div>
                ))}
              </div>
            ) : null}
            {approval.commands.length > 0 ? (
              <div className="chat-approval-cmds">
                <div className="chat-approval-cmds-title">待审批执行的命令：</div>
                {approval.commands.map((item, i) => (
                  <div key={`${message.id}-cmd-${i}`} className="chat-approval-action">
                    <div>
                      <b>{item.title || `命令 ${i + 1}`}</b>
                      <Tag color={riskColor(item.risk)} style={{ marginInlineStart: 8 }}>
                        {item.risk}
                      </Tag>
                    </div>
                    <pre className="chat-step-cmd">$ {item.cmd}</pre>
                  </div>
                ))}
              </div>
            ) : approval.actions.length > 0 ? (
              <div className="chat-approval-cmds">
                <div className="chat-approval-cmds-title">待审批动作：</div>
                {approval.actions.map((action, i) => (
                  <div key={`${message.id}-act-${i}`} className="chat-approval-action">
                    <div>
                      <b>{String(action.title ?? action.action_id ?? `action-${i + 1}`)}</b>
                      {action.risk ? (
                        <Tag color={riskColor(String(action.risk))} style={{ marginInlineStart: 8 }}>
                          {String(action.risk)}
                        </Tag>
                      ) : null}
                      {action.category ? (
                        <Tag style={{ marginInlineStart: 8 }}>{String(action.category)}</Tag>
                      ) : null}
                    </div>
                    {action.command ? (
                      <pre className="chat-step-cmd">$ {String(action.command)}</pre>
                    ) : null}
                    {action.description ? (
                      <div className="chat-step-summary">{String(action.description)}</div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
            <div className="chat-approval-btns">
              <Button
                type="primary"
                size="small"
                loading={approvalSubmitting}
                onClick={() => onApproval(message.id, true)}
              >
                批准并继续
              </Button>
              <Button
                danger
                size="small"
                loading={approvalSubmitting}
                onClick={() => onApproval(message.id, false)}
              >
                拒绝并停止
              </Button>
            </div>
          </div>
        ) : null}

        {!isUser && modelOutput ? (
          <div className="chat-mini-bubble chat-mini-bubble-conclusion">
            <div className="chat-mini-title">
              <span>模型输出</span>
              {isStreaming ? (
                <span className="chat-streaming-dot" aria-label="streaming">
                  <LoadingOutlined style={{ fontSize: 12, marginInlineStart: 6 }} />
                </span>
              ) : null}
            </div>
            <div className="chat-conclusion-text">
              <div style={{ whiteSpace: 'pre-wrap' }}>
                {streamedOutput}
                {isStreaming ? <span className="chat-caret">▍</span> : null}
              </div>
            </div>
          </div>
        ) : null}

        {recommendations.length > 0 && !modelOutput ? (
          <div className="chat-subcard">
            <div className="chat-subcard-title">处置建议</div>
            {recommendations.map((rec: Record<string, unknown>, i: number) => (
              <div key={`${message.id}-rec-${i}`} className="chat-subcard-item">
                {normalizePlainText(
                  String(rec.title ?? rec.summary ?? rec.description ?? JSON.stringify(rec)),
                )}
              </div>
            ))}
          </div>
        ) : null}

        {citations.length > 0 ? (
          <div className="chat-subcard">
            <div className="chat-subcard-title">引用来源 ({citations.length})</div>
            {citations.map((cite: string, i: number) => (
              <div key={`${message.id}-cite-${i}`} className="chat-subcard-item chat-cite">
                [{i + 1}] {cite}
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}
