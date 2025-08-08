'use client'

import { useRouter } from 'next/navigation'
import VoiceInterface from './VoiceInterface'

export default function VoiceProvider() {
  const router = useRouter()

  const handleActionTrigger = (action: string, data?: any) => {
    switch (action) {
      case 'navigate':
        router.push(data)
        break
      
      case 'assessment':
        console.log(`Starting ${data.type} assessment:`, data.message)
        // Show assessment modal or redirect to assessment page
        if (data.type === 'soc') {
          router.push('/dashboard?module=policies&assessment=soc')
        } else if (data.type === 'security') {
          router.push('/dashboard?module=network&assessment=security')
        } else if (data.type === 'cost') {
          router.push('/dashboard?module=costs&assessment=cost')
        } else if (data.type === 'rbac') {
          router.push('/dashboard?module=rbac&assessment=rbac')
        } else {
          router.push('/dashboard')
        }
        break
      
      case 'analysis':
        console.log(`Starting ${data} analysis`)
        router.push(`/dashboard?module=${data}&analysis=true`)
        break
      
      case 'emergency':
        console.log(`Emergency ${data} alert triggered`)
        // Trigger emergency protocols
        router.push(`/dashboard?alert=${data}`)
        break
      
      case 'ai_suggestions':
        console.log('AI Suggestions:', data)
        // Show suggestions in UI
        break
      
      case 'action':
        console.log(`Action triggered: ${data}`)
        if (data === 'show_recommendations') {
          router.push('/dashboard?focus=recommendations')
        }
        break
      
      default:
        console.log('Unknown action:', action, data)
    }
  }

  return <VoiceInterface onActionTrigger={handleActionTrigger} />
}