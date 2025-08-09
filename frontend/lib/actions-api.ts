export interface CreateActionRequest {
  action_type: string
  resource_id?: string
  params?: Record<string, any>
}

export interface ActionRecord {
  id: string
  action_type: string
  resource_id?: string
  status: string
  params: Record<string, any>
  result?: any
  created_at?: string
  updated_at?: string
}

// Use relative URLs to leverage Next.js proxy configuration
const API_BASE = ''

export async function createAction(payload: CreateActionRequest): Promise<{ action_id: string }> {
  const res = await fetch(`/api/v1/actions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!res.ok) throw new Error(`Failed to create action: ${res.status}`)
  return res.json()
}

export async function getAction(actionId: string): Promise<ActionRecord> {
  const res = await fetch(`/api/v1/actions/${actionId}`)
  if (!res.ok) throw new Error(`Failed to get action: ${res.status}`)
  return res.json()
}

export function streamActionEvents(actionId: string, onEvent: (msg: string) => void): EventSource {
  const es = new EventSource(`/api/v1/actions/${actionId}/events`)
  es.onmessage = (ev) => {
    onEvent(ev.data)
  }
  return es
}


