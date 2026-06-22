import { useMemo } from 'react'
import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import {
  App as AntApp,
  Layout,
  Menu,
  Space,
  Tag,
  Typography,
} from 'antd'
import {
  CommentOutlined,
  DatabaseOutlined,
  HddOutlined,
  SettingOutlined,
  DesktopOutlined,
} from '@ant-design/icons'
import { ChatPage } from './pages/ChatPage'
import { ConfigPage } from './pages/ConfigPage'
import { KnowledgePage } from './pages/KnowledgePage'
import { MemoryPage } from './pages/MemoryPage'
import { SystemPage } from './pages/SystemPage'

const { Header, Sider, Content } = Layout

const navigationItems = [
  {
    key: '/chat',
    icon: <CommentOutlined />,
    label: <Link to="/chat">Chat</Link>,
  },
  {
    key: '/knowledge',
    icon: <DatabaseOutlined />,
    label: <Link to="/knowledge">Knowledge</Link>,
  },
  {
    key: '/memory',
    icon: <HddOutlined />,
    label: <Link to="/memory">Memory</Link>,
  },
  {
    key: '/config',
    icon: <SettingOutlined />,
    label: <Link to="/config">Config</Link>,
  },
  {
    key: '/system',
    icon: <DesktopOutlined />,
    label: <Link to="/system">System</Link>,
  },
]

const titleMap: Record<string, string> = {
  '/chat': 'Chat',
  '/knowledge': 'Knowledge Console',
  '/memory': 'Memory Console',
  '/config': 'Runtime Config',
  '/system': 'System Status',
}

function DashboardApp() {
  const location = useLocation()
  const selectedKey = navigationItems.some((item) => item.key === location.pathname)
    ? location.pathname
    : '/chat'

  const pageTitle = useMemo(
    () => titleMap[selectedKey] ?? 'Dashboard',
    [selectedKey],
  )

  const isChatPage = selectedKey === '/chat'

  return (
    <AntApp>
      <Layout className="app-shell">
        <Sider
          breakpoint="lg"
          collapsedWidth="0"
          width={220}
          className="app-sider"
        >
          <div className="app-logo">
            <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
              Ops RAG
            </Typography.Title>
            <Typography.Text style={{ color: 'rgba(255,255,255,0.65)', fontSize: 12 }}>
              Chat Dashboard
            </Typography.Text>
          </div>
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={[selectedKey]}
            items={navigationItems}
          />
        </Sider>
        <Layout>
          {isChatPage ? null : (
            <Header className="app-header">
              <Space className="header-space" align="center">
                <div>
                  <Typography.Title level={3} style={{ margin: 0 }}>
                    {pageTitle}
                  </Typography.Title>
                  <Typography.Text type="secondary">
                    统一走 `/api`，支持本地代理与同域反向代理
                  </Typography.Text>
                </div>
                <Tag color="processing">MVP</Tag>
              </Space>
            </Header>
          )}
          <Content className={isChatPage ? 'app-content' : 'app-content app-content-scroll'} style={isChatPage ? { height: '100vh' } : undefined}>
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/knowledge" element={<KnowledgePage />} />
              <Route path="/memory" element={<MemoryPage />} />
              <Route path="/config" element={<ConfigPage />} />
              <Route path="/system" element={<SystemPage />} />
              <Route path="*" element={<Navigate to="/chat" replace />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </AntApp>
  )
}

export default function App() {
  return <DashboardApp />
}
