import { useState, useEffect } from 'react'
import { useMsal } from '@azure/msal-react'
import { InteractionStatus } from '@azure/msal-browser'

export const useAuthStatus = () => {
  const { inProgress } = useMsal()
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    // Set ready when MSAL is done with initialization
    if (inProgress === InteractionStatus.None) {
      setIsReady(true)
    }
  }, [inProgress])

  return {
    isReady,
    inProgress,
    isInitializing: inProgress === InteractionStatus.Startup,
    isLoggingIn: inProgress === InteractionStatus.Login,
    isLoggingOut: inProgress === InteractionStatus.Logout,
    isAcquiringToken: inProgress === InteractionStatus.AcquireToken,
  }
}