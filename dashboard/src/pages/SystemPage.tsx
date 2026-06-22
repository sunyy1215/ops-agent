import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { TableColumnsType } from 'antd'
import { CloudServerOutlined, LinkOutlined, ReloadOutlined } from '@ant-design/icons'
import { API_PREFIX, apiClient } from '../api/client'
import type {
  HealthzResponse,
  MetricsSummaryResponse,
  RunStateResponse,
  SessionSummary,
  SessionsResponse,
  SkillCatalogItem,
  SkillsCatalogResponse,
} from '../types'

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

function getStatusColor(status?: string | null) {
  switch ((status ?? '').toLowerCase()) {
    case 'ok':
    case 'completed':
    case 'passed':
    case 'approved':
      return 'success'
    case 'running':
    case 'pending':
    case 'waiting_approval':
    case 'mixed':
      return 'processing'
    case 'failed':
    case 'rejected':
      return 'error'
    default:
      return 'default'
  }
}

function formatTimestamp(value?: string) {
  if (!value) {
    return '-'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString('zh-CN', { hour12: false })
}

function renderCountTags(counts: Record<string, number>) {
  const entries = Object.entries(counts)
  if (entries.length === 0) {
    return <Typography.Text type="secondary">暂无数据</Typography.Text>
  }

  return (
    <Space size={[8, 8]} wrap>
      {entries.map(([key, value]) => (
        <Tag key={key}>
          {key}: {value}
        </Tag>
      ))}
    </Space>
  )
}

export function SystemPage() {
  const [healthz, setHealthz] = useState<HealthzResponse>()
  const [sessions, setSessions] = useState<SessionsResponse>()
  const [skillsCatalog, setSkillsCatalog] = useState<SkillsCatalogResponse>()
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummaryResponse>()
  const [runState, setRunState] = useState<RunStateResponse>()
  const [selectedThreadId, setSelectedThreadId] = useState<string>()
  const [loading, setLoading] = useState(true)
  const [runStateLoading, setRunStateLoading] = useState(false)
  const [error, setError] = useState<string>()
  const [runStateError, setRunStateError] = useState<string>()

  const loadDashboard = useCallback(async () => {
    setLoading(true)
    setError(undefined)
    try {
      const [healthzResponse, sessionsResponse, skillsResponse, metricsResponse] =
        await Promise.all([
          apiClient.getHealthz(),
          apiClient.getSessions(20),
          apiClient.getSkillsCatalog(),
          apiClient.getMetricsSummary(),
        ])

      setHealthz(healthzResponse)
      setSessions(sessionsResponse)
      setSkillsCatalog(skillsResponse)
      setMetricsSummary(metricsResponse)

      const nextThreadId =
        selectedThreadId &&
        sessionsResponse.items.some((item) => item.thread_id === selectedThreadId)
          ? selectedThreadId
          : sessionsResponse.items[0]?.thread_id

      setSelectedThreadId(nextThreadId)

      if (nextThreadId) {
        setRunStateLoading(true)
        setRunStateError(undefined)
        try {
          const runStateResponse = await apiClient.getRunState(nextThreadId)
          setRunState(runStateResponse)
        } catch (err) {
          setRunState(undefined)
          setRunStateError(err instanceof Error ? err.message : '获取运行态失败')
        } finally {
          setRunStateLoading(false)
        }
      } else {
        setRunState(undefined)
        setRunStateError(undefined)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取系统状态失败')
    } finally {
      setLoading(false)
    }
  }, [selectedThreadId])

  const loadRunState = useCallback(async (threadId: string) => {
    setRunStateLoading(true)
    setRunStateError(undefined)
    try {
      const response = await apiClient.getRunState(threadId)
      setRunState(response)
    } catch (err) {
      setRunState(undefined)
      setRunStateError(err instanceof Error ? err.message : '获取运行态失败')
    } finally {
      setRunStateLoading(false)
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadDashboard()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [loadDashboard])

  const handleSelectThread = useCallback(
    (threadId: string) => {
      setSelectedThreadId(threadId)
      void loadRunState(threadId)
    },
    [loadRunState],
  )

  const sessionColumns = useMemo<TableColumnsType<SessionSummary>>(
    () => [
      {
        title: 'Thread',
        dataIndex: 'thread_id',
        key: 'thread_id',
        width: 160,
      },
      {
        title: 'Workflow',
        dataIndex: 'workflow_status',
        key: 'workflow_status',
        width: 140,
        render: (value: SessionSummary['workflow_status']) => (
          <Tag color={getStatusColor(value)}>{value || 'unknown'}</Tag>
        ),
      },
      {
        title: 'Task',
        dataIndex: 'task_status_suggestion',
        key: 'task_status_suggestion',
        width: 180,
        render: (value: SessionSummary['task_status_suggestion']) => (
          <Space direction="vertical" size={0}>
            <Tag color={getStatusColor(value.code)}>{value.code}</Tag>
            <Typography.Text type="secondary">{value.summary}</Typography.Text>
          </Space>
        ),
      },
      {
        title: 'Verification',
        dataIndex: 'verification_summary',
        key: 'verification_summary',
        width: 150,
        render: (value: SessionSummary['verification_summary']) => (
          <Tag color={getStatusColor(value.status)}>{value.status}</Tag>
        ),
      },
      {
        title: 'Updated',
        dataIndex: 'updated_at',
        key: 'updated_at',
        width: 180,
        render: (value: string) => formatTimestamp(value),
      },
    ],
    [],
  )

  const skillColumns = useMemo<TableColumnsType<SkillCatalogItem>>(
    () => [
      {
        title: 'Skill ID',
        dataIndex: 'skill_id',
        key: 'skill_id',
      },
      {
        title: 'Kind',
        dataIndex: 'kind',
        key: 'kind',
        width: 140,
        render: (value: SkillCatalogItem['kind']) => (
          <Tag color={value === 'complex_dev' ? 'purple' : 'blue'}>{String(value ?? '-')}</Tag>
        ),
      },
    ],
    [],
  )

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {error ? <Alert type="error" message={error} showIcon /> : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card
            title="服务健康状态"
            extra={
              <Button icon={<ReloadOutlined />} onClick={() => void loadDashboard()}>
                刷新系统面板
              </Button>
            }
          >
            <Spin spinning={loading}>
              {healthz ? (
                <Descriptions column={1} bordered size="small">
                  <Descriptions.Item label="status">
                    <Tag color={healthz.status === 'ok' ? 'success' : 'error'}>
                      {healthz.status}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="app_name">
                    {healthz.app_name}
                  </Descriptions.Item>
                  <Descriptions.Item label="environment">
                    {healthz.environment}
                  </Descriptions.Item>
                </Descriptions>
              ) : (
                <Typography.Text type="secondary">
                  等待返回健康检查结果
                </Typography.Text>
              )}
            </Spin>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Card title="接入方式">
              <Space direction="vertical" size="small">
                <Typography.Text>
                  <LinkOutlined /> 前端统一通过 `{API_PREFIX}` 访问 API
                </Typography.Text>
                <Typography.Text>
                  <CloudServerOutlined /> 开发态由 `vite.config.ts` 代理到 `http://127.0.0.1:8000`
                </Typography.Text>
                <Typography.Text>
                  生产态可复用同域 `/api` 反向代理，避免改动页面代码
                </Typography.Text>
              </Space>
            </Card>
            <Card title="联调摘要">
              <Space size={[8, 8]} wrap>
                <Tag color="blue">`GET /healthz`</Tag>
                <Tag color="cyan">`GET /sessions`</Tag>
                <Tag color="geekblue">`GET /runs/{'{thread_id}'}/state`</Tag>
                <Tag color="purple">`GET /skills/catalog`</Tag>
                <Tag color="gold">`GET /metrics/summary`</Tag>
              </Space>
              <Typography.Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
                选中左下方会话后，右侧会自动加载对应运行态，便于定位审批、执行和验证阶段的当前状态。
              </Typography.Paragraph>
            </Card>
          </Space>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic title="Sessions" value={metricsSummary?.totals.sessions ?? 0} />
          </Card>
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic title="Checkpoints" value={metricsSummary?.totals.checkpoints ?? 0} />
          </Card>
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic title="Skills" value={metricsSummary?.totals.skills ?? 0} />
          </Card>
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic title="Pending Approvals" value={metricsSummary?.totals.pending_approvals ?? 0} />
          </Card>
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic
              title="Executed Calls"
              value={metricsSummary?.totals.executed_skill_calls ?? 0}
            />
          </Card>
        </Col>
        <Col xs={12} md={8} xl={4}>
          <Card>
            <Statistic
              title="Failed Calls"
              value={metricsSummary?.totals.failed_skill_calls ?? 0}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={14}>
          <Card
            title="Sessions"
            extra={
              <Typography.Text type="secondary">
                共 {sessions?.count ?? 0} 条
              </Typography.Text>
            }
          >
            <Spin spinning={loading}>
              <Table<SessionSummary>
                rowKey="thread_id"
                columns={sessionColumns}
                dataSource={sessions?.items ?? []}
                pagination={false}
                size="small"
                scroll={{ x: 860 }}
                locale={{ emptyText: <Empty description="暂无会话" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
                rowSelection={{
                  type: 'radio',
                  selectedRowKeys: selectedThreadId ? [selectedThreadId] : [],
                  onChange: (selectedRowKeys) => {
                    const nextThreadId = selectedRowKeys[0]
                    if (typeof nextThreadId === 'string') {
                      handleSelectThread(nextThreadId)
                    }
                  },
                }}
                onRow={(record) => ({
                  onClick: () => handleSelectThread(record.thread_id),
                })}
              />
            </Spin>
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card
            title="Run State"
            extra={
              selectedThreadId ? (
                <Button loading={runStateLoading} onClick={() => void loadRunState(selectedThreadId)}>
                  刷新当前线程
                </Button>
              ) : null
            }
          >
            {runStateError ? (
              <Alert
                type="error"
                showIcon
                message={runStateError}
                style={{ marginBottom: 16 }}
              />
            ) : null}
            <Spin spinning={runStateLoading}>
              {runState ? (
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <Descriptions column={1} bordered size="small">
                    <Descriptions.Item label="thread_id">
                      {runState.thread_id}
                    </Descriptions.Item>
                    <Descriptions.Item label="checkpoint_id">
                      {runState.checkpoint_id || '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label="route">
                      <Tag>{runState.route || 'unknown'}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="workflow_status">
                      <Tag color={getStatusColor(runState.workflow_status)}>
                        {runState.workflow_status || 'unknown'}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="approval_status">
                      <Tag color={getStatusColor(runState.approval_status)}>
                        {runState.approval_status || 'not_set'}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="next_nodes">
                      {runState.next_nodes.length > 0 ? runState.next_nodes.join(', ') : '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label="task_status">
                      <Space direction="vertical" size={0}>
                        <Tag color={getStatusColor(runState.task_status_suggestion.code)}>
                          {runState.task_status_suggestion.code}
                        </Tag>
                        <Typography.Text type="secondary">
                          {runState.task_status_suggestion.summary}
                        </Typography.Text>
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="verification">
                      <Space direction="vertical" size={0}>
                        <Tag color={getStatusColor(runState.verification_summary.status)}>
                          {runState.verification_summary.status}
                        </Tag>
                        <Typography.Text type="secondary">
                          {runState.verification_summary.summary}
                        </Typography.Text>
                      </Space>
                    </Descriptions.Item>
                  </Descriptions>
                  <Card type="inner" title="Pending Interrupts">
                    <JsonBlock value={runState.pending_interrupts} />
                  </Card>
                  <Card type="inner" title="Raw State">
                    <JsonBlock value={runState.state} />
                  </Card>
                </Space>
              ) : (
                <Empty description="请选择会话以查看运行态" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              )}
            </Spin>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={12}>
          <Card title="Skills Catalog">
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Space size={[8, 8]} wrap>
                <Tag color="blue">all: {skillsCatalog?.counts.all ?? 0}</Tag>
                <Tag color="cyan">regular: {skillsCatalog?.counts.regular ?? 0}</Tag>
                <Tag color="purple">complex_dev: {skillsCatalog?.counts.complex_dev ?? 0}</Tag>
              </Space>
              <div>
                <Typography.Text type="secondary">允许业务域</Typography.Text>
                <div style={{ marginTop: 8 }}>
                  {skillsCatalog && skillsCatalog.allowed_business_domains.length > 0 ? (
                    <Space size={[8, 8]} wrap>
                      {skillsCatalog.allowed_business_domains.map((domain) => (
                        <Tag key={domain}>{domain}</Tag>
                      ))}
                    </Space>
                  ) : (
                    <Typography.Text type="secondary">未配置过滤，默认展示全部业务域</Typography.Text>
                  )}
                </div>
              </div>
              <Table<SkillCatalogItem>
                rowKey="skill_id"
                columns={skillColumns}
                dataSource={skillsCatalog?.groups.all ?? []}
                pagination={false}
                size="small"
                locale={{ emptyText: <Empty description="暂无技能目录" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
              />
            </Space>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="Metrics Summary">
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="last_updated_at">
                  {formatTimestamp(metricsSummary?.last_updated_at)}
                </Descriptions.Item>
              </Descriptions>
              <div>
                <Typography.Text type="secondary">workflow_status_counts</Typography.Text>
                <div style={{ marginTop: 8 }}>
                  {renderCountTags(metricsSummary?.workflow_status_counts ?? {})}
                </div>
              </div>
              <div>
                <Typography.Text type="secondary">route_counts</Typography.Text>
                <div style={{ marginTop: 8 }}>
                  {renderCountTags(metricsSummary?.route_counts ?? {})}
                </div>
              </div>
              <div>
                <Typography.Text type="secondary">task_status_suggestion_counts</Typography.Text>
                <div style={{ marginTop: 8 }}>
                  {renderCountTags(metricsSummary?.task_status_suggestion_counts ?? {})}
                </div>
              </div>
              <div>
                <Typography.Text type="secondary">verification_status_counts</Typography.Text>
                <div style={{ marginTop: 8 }}>
                  {renderCountTags(metricsSummary?.verification_status_counts ?? {})}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </Space>
  )
}
