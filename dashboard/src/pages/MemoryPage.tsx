import { useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import { apiClient } from '../api/client'
import type { MemoryRecallResponse, MemoryWriteResponse } from '../types'

type MemoryWriteFormValues = {
  memory_id: string
  memory_type: string
  user_id?: string
  session_id?: string
  content: string
  tagsText?: string
  metadataText?: string
}

type MemoryRecallFormValues = {
  query: string
  user_id?: string
  top_k?: number
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

export function MemoryPage() {
  const [defaultMemoryId] = useState(() => `memory-${Date.now()}`)
  const [writeResult, setWriteResult] = useState<MemoryWriteResponse>()
  const [recallResult, setRecallResult] = useState<MemoryRecallResponse>()
  const [loadingWrite, setLoadingWrite] = useState(false)
  const [loadingRecall, setLoadingRecall] = useState(false)
  const [error, setError] = useState<string>()

  const handleWrite = async (values: MemoryWriteFormValues) => {
    setLoadingWrite(true)
    setError(undefined)
    try {
      const response = await apiClient.writeMemory({
        records: [
          {
            memory_id: values.memory_id.trim(),
            memory_type: values.memory_type.trim(),
            user_id: values.user_id?.trim() || undefined,
            session_id: values.session_id?.trim() || undefined,
            content: values.content.trim(),
            tags: values.tagsText
              ?.split(',')
              .map((item) => item.trim())
              .filter(Boolean),
            metadata: values.metadataText?.trim()
              ? (JSON.parse(values.metadataText) as Record<string, unknown>)
              : {},
          },
        ],
      })
      setWriteResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆写入失败')
    } finally {
      setLoadingWrite(false)
    }
  }

  const handleRecall = async (values: MemoryRecallFormValues) => {
    setLoadingRecall(true)
    setError(undefined)
    try {
      const response = await apiClient.recallMemory({
        query: values.query.trim(),
        user_id: values.user_id?.trim() || undefined,
        top_k: values.top_k ?? 3,
      })
      setRecallResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆召回失败')
    } finally {
      setLoadingRecall(false)
    }
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {error ? <Alert type="error" message={error} showIcon /> : null}

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={12}>
          <Card title="记忆写入">
            <Form
              layout="vertical"
              onFinish={handleWrite}
              initialValues={{
                memory_id: defaultMemoryId,
                memory_type: 'note',
                metadataText: '{"source":"dashboard"}',
              }}
            >
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item
                    label="Memory ID"
                    name="memory_id"
                    rules={[{ required: true, message: '请输入 memory_id' }]}
                  >
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="Memory Type"
                    name="memory_type"
                    rules={[{ required: true, message: '请输入 memory_type' }]}
                  >
                    <Input placeholder="incident / note / preference" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item label="User ID" name="user_id">
                    <Input />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Session ID" name="session_id">
                    <Input />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item
                label="Content"
                name="content"
                rules={[{ required: true, message: '请输入内容' }]}
              >
                <Input.TextArea autoSize={{ minRows: 4, maxRows: 8 }} />
              </Form.Item>
              <Form.Item label="Tags" name="tagsText">
                <Input placeholder="ops,incident" />
              </Form.Item>
              <Form.Item label="Metadata JSON" name="metadataText">
                <Input.TextArea autoSize={{ minRows: 3, maxRows: 6 }} />
              </Form.Item>
              <Button htmlType="submit" type="primary" loading={loadingWrite}>
                调用 `POST /memory/write`
              </Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="记忆召回">
            <Form layout="vertical" onFinish={handleRecall} initialValues={{ top_k: 3 }}>
              <Form.Item
                label="Query"
                name="query"
                rules={[{ required: true, message: '请输入 query' }]}
              >
                <Input placeholder="例如：redis timeout" />
              </Form.Item>
              <Form.Item label="User ID" name="user_id">
                <Input />
              </Form.Item>
              <Form.Item label="Top K" name="top_k">
                <InputNumber min={1} max={20} style={{ width: '100%' }} />
              </Form.Item>
              <Button htmlType="submit" type="primary" loading={loadingRecall}>
                调用 `POST /memory/recall`
              </Button>
            </Form>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={10}>
          <Card title="写入结果">
            {writeResult ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Typography.Text>count: {writeResult.count}</Typography.Text>
                <Typography.Text>
                  written_ids:
                  {writeResult.written_ids.map((item) => (
                    <Tag style={{ marginInlineStart: 8 }} color="blue" key={item}>
                      {item}
                    </Tag>
                  ))}
                </Typography.Text>
                <JsonBlock value={writeResult} />
              </Space>
            ) : (
              <Typography.Text type="secondary">
                写入后将在这里展示数量与写入 ID
              </Typography.Text>
            )}
          </Card>
        </Col>
        <Col xs={24} xl={14}>
          <Card title="召回结果">
            {recallResult ? (
              <Table
                rowKey={(record: Record<string, unknown>) => String(record.memory_id)}
                pagination={false}
                size="small"
                dataSource={recallResult.results}
                scroll={{ x: 900 }}
                columns={[
                  {
                    title: 'memory_id',
                    dataIndex: 'memory_id',
                  },
                  {
                    title: 'memory_type',
                    dataIndex: 'memory_type',
                  },
                  {
                    title: 'score',
                    dataIndex: 'score',
                    render: (value: unknown) => String(value ?? '-'),
                  },
                  {
                    title: 'tags',
                    dataIndex: 'tags',
                    render: (value: unknown) =>
                      Array.isArray(value)
                        ? value.map((item) => (
                            <Tag color="processing" key={String(item)}>
                              {String(item)}
                            </Tag>
                          ))
                        : '-',
                  },
                  {
                    title: 'content',
                    dataIndex: 'content',
                    width: 360,
                    render: (value: unknown) => (
                      <Typography.Paragraph ellipsis={{ rows: 3, expandable: true, symbol: '展开' }}>
                        {String(value ?? '')}
                      </Typography.Paragraph>
                    ),
                  },
                ]}
              />
            ) : (
              <Typography.Text type="secondary">
                召回结果会展示标签、元数据和分数
              </Typography.Text>
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  )
}
