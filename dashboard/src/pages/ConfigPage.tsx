import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Row,
  Space,
  Spin,
  Switch,
  Tag,
  Typography,
} from 'antd'
import { Link } from 'react-router-dom'
import { apiClient } from '../api/client'
import type { HealthzResponse, PublicConfigResponse } from '../types'

type ConfigFormValues = Record<string, boolean | number | string | null>

type PrimitiveValue = boolean | number | string | null

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

function flattenSections(config: PublicConfigResponse): Record<string, PrimitiveValue> {
  const values: Record<string, PrimitiveValue> = {}
  Object.values(config.sections).forEach((section) => {
    Object.entries(section).forEach(([field, value]) => {
      if (
        typeof value === 'boolean' ||
        typeof value === 'number' ||
        typeof value === 'string' ||
        value === null
      ) {
        values[field] = value
      }
    })
  })
  return values
}

function normalizeValue(value: PrimitiveValue, previousValue: PrimitiveValue): PrimitiveValue {
  if (typeof previousValue === 'number') {
    return typeof value === 'number' ? value : Number(value ?? 0)
  }
  if (typeof previousValue === 'boolean') {
    return Boolean(value)
  }
  if (value === null) {
    return null
  }
  return String(value ?? '').trim()
}

function renderFieldInput(field: string, value: PrimitiveValue) {
  if (typeof value === 'boolean') {
    return (
      <Form.Item label={field} name={field} valuePropName="checked">
        <Switch />
      </Form.Item>
    )
  }

  if (typeof value === 'number') {
    return (
      <Form.Item label={field} name={field}>
        <InputNumber style={{ width: '100%' }} />
      </Form.Item>
    )
  }

  return (
    <Form.Item label={field} name={field}>
      <Input allowClear />
    </Form.Item>
  )
}

export function ConfigPage() {
  const [form] = Form.useForm<ConfigFormValues>()
  const [config, setConfig] = useState<PublicConfigResponse>()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string>()
  const [saveInfo, setSaveInfo] = useState<string>()
  const [saveWarnings, setSaveWarnings] = useState<Array<{ field: string; reason: string }>>([])
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionInfo, setConnectionInfo] = useState<HealthzResponse>()
  const [connectionError, setConnectionError] = useState<string>()
  const [connectionCheckedAt, setConnectionCheckedAt] = useState<string>()

  const loadConfig = useCallback(async () => {
    setLoading(true)
    setError(undefined)
    try {
      const response = await apiClient.getPublicConfig()
      setConfig(response)
      form.setFieldsValue(flattenSections(response))
    } catch (err) {
      setError(err instanceof Error ? err.message : '读取公开配置失败')
    } finally {
      setLoading(false)
    }
  }, [form])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadConfig()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [loadConfig])

  const editableFieldSet = useMemo(
    () => new Set(config?.editable_fields ?? []),
    [config],
  )

  const flattenedValues = useMemo(
    () => (config ? flattenSections(config) : {}),
    [config],
  )

  const handleSave = async (values: ConfigFormValues) => {
    if (!config) {
      return
    }

    setSaving(true)
    setError(undefined)
    setSaveInfo(undefined)
    setSaveWarnings([])

    try {
      const updates = Object.fromEntries(
        Object.entries(values)
          .filter(([field]) => editableFieldSet.has(field))
          .map(([field, value]) => [
            field,
            normalizeValue(
              (value as PrimitiveValue) ?? null,
              flattenedValues[field] ?? null,
            ),
          ]),
      )

      const response = await apiClient.updateRuntimeConfig({ updates })
      setConfig(response.public_config)
      form.setFieldsValue(flattenSections(response.public_config))
      setSaveWarnings(response.rejected)
      setSaveInfo(`本次成功应用 ${response.applied_count} 个字段`)
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存运行时配置失败')
    } finally {
      setSaving(false)
    }
  }

  const handleTestConnection = async () => {
    setTestingConnection(true)
    setConnectionError(undefined)
    try {
      const response = await apiClient.getHealthz()
      setConnectionInfo(response)
      setConnectionCheckedAt(new Date().toLocaleString('zh-CN'))
    } catch (err) {
      setConnectionInfo(undefined)
      setConnectionError(err instanceof Error ? err.message : '测试连接失败')
      setConnectionCheckedAt(new Date().toLocaleString('zh-CN'))
    } finally {
      setTestingConnection(false)
    }
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="配置页仅展示公开配置，并只对白名单字段开放热更新。"
        description="敏感字段不会出现在可编辑区域中，后端仍会拒绝任何越权或敏感修改。"
      />

      {error ? <Alert type="error" message={error} showIcon /> : null}
      {saveInfo ? <Alert type="success" message={saveInfo} showIcon /> : null}
      {saveWarnings.length > 0 ? (
        <Alert
          type="warning"
          showIcon
          message="部分字段未被应用"
          description={saveWarnings.map((item) => `${item.field}: ${item.reason}`).join(' | ')}
        />
      ) : null}

      <Spin spinning={loading}>
        <Row gutter={[16, 16]} align="top">
          <Col xs={24} xl={16}>
            <Card
              title="运行时配置"
              extra={
                <Space>
                  <Button onClick={() => void loadConfig()}>刷新</Button>
                  <Button onClick={() => form.resetFields()}>重置表单</Button>
                </Space>
              }
            >
              <Form form={form} layout="vertical" onFinish={handleSave}>
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {config
                    ? Object.entries(config.sections)
                        .filter(([section]) => section !== 'secrets')
                        .map(([section, fields]) => (
                          <Card
                            key={section}
                            type="inner"
                            title={section.toUpperCase()}
                            extra={
                              <Tag color="blue">
                                {Object.keys(fields).filter((field) => editableFieldSet.has(field)).length}{' '}
                                editable
                              </Tag>
                            }
                          >
                            <Row gutter={12}>
                              {Object.entries(fields)
                                .filter(([field, value]) => editableFieldSet.has(field) && !Array.isArray(value))
                                .map(([field, value]) => (
                                  <Col xs={24} md={12} key={field}>
                                    {renderFieldInput(field, (value as PrimitiveValue) ?? null)}
                                  </Col>
                                ))}
                            </Row>
                            {Object.keys(fields).every((field) => !editableFieldSet.has(field)) ? (
                              <Typography.Text type="secondary">
                                当前分组没有可热更新字段，右侧展示为只读摘要。
                              </Typography.Text>
                            ) : null}
                          </Card>
                        ))
                    : null}
                </Space>
                <Space style={{ marginTop: 16 }}>
                  <Button htmlType="submit" type="primary" loading={saving}>
                    保存到 `PUT /config/runtime`
                  </Button>
                  <Button onClick={() => void loadConfig()}>
                    重新加载 `GET /config/public`
                  </Button>
                </Space>
              </Form>
            </Card>
          </Col>

          <Col xs={24} xl={8}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Card title="公开配置摘要">
                {config ? (
                  <JsonBlock value={config.sections} />
                ) : (
                  <Typography.Text type="secondary">等待配置加载</Typography.Text>
                )}
              </Card>
              <Card title="安全说明">
                <Typography.Paragraph>
                  后端只返回 `secrets.*_configured` 这类布尔状态，不会返回真实密钥值。
                </Typography.Paragraph>
                <Typography.Paragraph>
                  前端保存时仅提交 `editable_fields` 白名单字段，默认不发送任何敏感键名。
                </Typography.Paragraph>
                <Typography.Paragraph style={{ marginBottom: 0 }}>
                  若需要变更密钥或凭据，请改用环境变量、SecretRef 或受限运维通道。
                </Typography.Paragraph>
              </Card>
              <Card
                title="测试连接"
                extra={
                  <Button loading={testingConnection} onClick={() => void handleTestConnection()}>
                    测试 `GET /healthz`
                  </Button>
                }
              >
                <Typography.Paragraph>
                  用于确认 Dashboard 当前代理配置和后端 API 是否可达，不会读取或提交任何敏感配置。
                </Typography.Paragraph>
                {connectionError ? (
                  <Alert
                    type="error"
                    showIcon
                    message="连接失败"
                    description={connectionError}
                    style={{ marginBottom: 12 }}
                  />
                ) : null}
                {connectionInfo ? (
                  <Descriptions column={1} bordered size="small">
                    <Descriptions.Item label="status">
                      <Tag color={connectionInfo.status === 'ok' ? 'success' : 'error'}>
                        {connectionInfo.status}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="app_name">
                      {connectionInfo.app_name}
                    </Descriptions.Item>
                    <Descriptions.Item label="environment">
                      {connectionInfo.environment}
                    </Descriptions.Item>
                    <Descriptions.Item label="checked_at">
                      {connectionCheckedAt ?? '-'}
                    </Descriptions.Item>
                  </Descriptions>
                ) : (
                  <Typography.Text type="secondary">
                    尚未执行连接测试。
                  </Typography.Text>
                )}
                <Typography.Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
                  需要查看更多运行态诊断时，可前往 <Link to="/system">System 页面</Link> 查看会话、技能目录和指标摘要。
                </Typography.Paragraph>
              </Card>
              <Card title="编辑白名单">
                {config ? (
                  <Space size={[4, 8]} wrap>
                    {config.editable_fields.map((field) => (
                      <Tag key={field}>{field}</Tag>
                    ))}
                  </Space>
                ) : null}
              </Card>
            </Space>
          </Col>
        </Row>
      </Spin>
    </Space>
  )
}
