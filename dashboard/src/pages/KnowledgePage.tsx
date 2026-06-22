import { useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  Row,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
} from 'antd'
import { apiClient } from '../api/client'
import type { KbIngestResponse, KbSearchResponse } from '../types'

type IngestFormValues = {
  collection?: string
  dry_run: boolean
  file_path?: string
  directory_path?: string
  glob?: string
  docsText?: string
}

type SearchFormValues = {
  query: string
  visibility?: string
  biz_domain?: string
  lang?: string
  tagsText?: string
  fused_top_k?: number
  rerank_top_k?: number
}

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

export function KnowledgePage() {
  const [ingestResult, setIngestResult] = useState<KbIngestResponse>()
  const [searchResult, setSearchResult] = useState<KbSearchResponse>()
  const [loadingIngest, setLoadingIngest] = useState(false)
  const [loadingSearch, setLoadingSearch] = useState(false)
  const [error, setError] = useState<string>()

  const handleIngest = async (values: IngestFormValues) => {
    setLoadingIngest(true)
    setError(undefined)

    try {
      const docsText = values.docsText?.trim()
      let payload

      if (values.file_path?.trim()) {
        payload = {
          collection: values.collection?.trim() || undefined,
          file_path: values.file_path.trim(),
          dry_run: values.dry_run,
        }
      } else if (values.directory_path?.trim()) {
        payload = {
          collection: values.collection?.trim() || undefined,
          directory_path: values.directory_path.trim(),
          glob: values.glob?.trim() || '**/*',
          dry_run: values.dry_run,
        }
      } else if (docsText) {
        payload = {
          collection: values.collection?.trim() || undefined,
          dry_run: values.dry_run,
          docs: [
            {
              id: `dashboard-${Date.now()}`,
              text: docsText,
              metadata: {
                source: 'dashboard',
              },
            },
          ],
        }
      } else {
        throw new Error('请至少填写文件路径、目录路径或内联文档内容之一')
      }

      const response = await apiClient.ingestKnowledge(payload)
      setIngestResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : '知识库导入失败')
    } finally {
      setLoadingIngest(false)
    }
  }

  const handleSearch = async (values: SearchFormValues) => {
    setLoadingSearch(true)
    setError(undefined)
    try {
      const response = await apiClient.searchKnowledge({
        query: values.query.trim(),
        visibility: values.visibility?.trim() || undefined,
        biz_domain: values.biz_domain?.trim() || undefined,
        lang: values.lang?.trim() || undefined,
        fused_top_k: values.fused_top_k ?? undefined,
        rerank_top_k: values.rerank_top_k ?? undefined,
        tags: values.tagsText
          ?.split(',')
          .map((item) => item.trim())
          .filter(Boolean),
      })
      setSearchResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : '知识库检索失败')
    } finally {
      setLoadingSearch(false)
    }
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {error ? <Alert type="error" message={error} showIcon /> : null}

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={12}>
          <Card title="知识库导入">
            <Form layout="vertical" onFinish={handleIngest} initialValues={{ dry_run: true, glob: '**/*' }}>
              <Form.Item label="Collection" name="collection">
                <Input placeholder="可选，默认使用后端配置 collection" />
              </Form.Item>
              <Form.Item label="File Path" name="file_path">
                <Input placeholder="/absolute/path/to/doc.md" />
              </Form.Item>
              <Form.Item label="Directory Path" name="directory_path">
                <Input placeholder="/absolute/path/to/docs" />
              </Form.Item>
              <Form.Item label="Glob" name="glob">
                <Input placeholder="**/*" />
              </Form.Item>
              <Form.Item label="Inline Document Text" name="docsText">
                <Input.TextArea
                  autoSize={{ minRows: 4, maxRows: 8 }}
                  placeholder="也可以直接粘贴一段文档内容进行导入验证"
                />
              </Form.Item>
              <Form.Item label="Dry Run" name="dry_run" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Button htmlType="submit" type="primary" loading={loadingIngest}>
                调用 `POST /kb/ingest`
              </Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card title="知识库检索">
            <Form layout="vertical" onFinish={handleSearch}>
              <Form.Item
                label="Query"
                name="query"
                rules={[{ required: true, message: '请输入检索 query' }]}
              >
                <Input placeholder="例如：milvus hybrid search" />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item label="Visibility" name="visibility">
                    <Input placeholder="internal" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Biz Domain" name="biz_domain">
                    <Input placeholder="ops" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item label="Lang" name="lang">
                    <Input placeholder="zh / en" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Tags" name="tagsText">
                    <Input placeholder="k8s,milvus" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item label="Fused Top K" name="fused_top_k">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Rerank Top K" name="rerank_top_k">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              <Button htmlType="submit" type="primary" loading={loadingSearch}>
                调用 `POST /kb/search`
              </Button>
            </Form>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={12}>
          <Card title="导入结果">
            {ingestResult ? (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Typography.Text>
                  collection: <Tag color="blue">{ingestResult.collection}</Tag>
                </Typography.Text>
                <Typography.Text>inserted_count: {ingestResult.inserted_count}</Typography.Text>
                <Typography.Text>dry_run: {String(ingestResult.dry_run)}</Typography.Text>
                <Divider style={{ margin: '12px 0' }} />
                <Typography.Text strong>stats</Typography.Text>
                <JsonBlock value={ingestResult.stats} />
                <Typography.Text strong>duplicate_relations</Typography.Text>
                <JsonBlock value={ingestResult.duplicate_relations} />
              </Space>
            ) : (
              <Typography.Text type="secondary">
                提交导入后将在这里展示 `dry_run`、`stats` 与 `duplicate_relations`
              </Typography.Text>
            )}
          </Card>
        </Col>
        <Col xs={24} xl={12}>
          <Card title="检索结果">
            {searchResult ? (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Typography.Text>
                  candidate_count: {searchResult.candidate_count} / reranked_count:{' '}
                  {searchResult.reranked_count}
                </Typography.Text>
                <Typography.Text strong>citations</Typography.Text>
                <JsonBlock value={searchResult.citations} />
                <Typography.Text strong>results</Typography.Text>
                <Table
                  rowKey={(_: Record<string, unknown>, index?: number) => `${index ?? 0}`}
                  pagination={false}
                  size="small"
                  dataSource={searchResult.results}
                  scroll={{ x: 800 }}
                  columns={[
                    {
                      title: 'doc_id',
                      dataIndex: 'doc_id',
                      render: (value: unknown) => String(value ?? '-'),
                    },
                    {
                      title: 'score',
                      dataIndex: 'score',
                      render: (value: unknown) => String(value ?? '-'),
                    },
                    {
                      title: 'source',
                      dataIndex: 'source',
                      render: (value: unknown) => String(value ?? '-'),
                    },
                    {
                      title: 'text',
                      dataIndex: 'text',
                      width: 360,
                      render: (value: unknown) => (
                        <Typography.Paragraph ellipsis={{ rows: 3, expandable: true, symbol: '展开' }}>
                          {String(value ?? '')}
                        </Typography.Paragraph>
                      ),
                    },
                  ]}
                />
              </Space>
            ) : (
              <Typography.Text type="secondary">
                检索结果会展示 `results` 与 `citations`
              </Typography.Text>
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  )
}
