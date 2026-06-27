import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ClerkProvider } from '@clerk/react'
import './index.css'
import App from './App.jsx'

const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

function MissingClerkConfig() {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'grid',
      placeItems: 'center',
      background: '#0D0D0D',
      color: '#ECECEC',
      fontFamily: 'Inter, Roboto, Helvetica, Arial, sans-serif',
      padding: 24,
      textAlign: 'center',
    }}>
      <div>
        <h1 style={{ fontSize: 24, marginBottom: 8 }}>Clerk is not configured</h1>
        <p style={{ color: '#B4B4B4', margin: 0 }}>
          Add VITE_CLERK_PUBLISHABLE_KEY to frontend/.env and restart the dev server.
        </p>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {clerkPublishableKey ? (
      <ClerkProvider publishableKey={clerkPublishableKey} afterSignOutUrl="/">
        <App />
      </ClerkProvider>
    ) : (
      <MissingClerkConfig />
    )}
  </StrictMode>,
)
