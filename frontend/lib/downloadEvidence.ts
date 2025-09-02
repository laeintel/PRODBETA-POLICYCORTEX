export function downloadEvidence(
  rows: {
    hash?: string | null
    merkle_root?: string | null
    signature?: string | null
    [key: string]: any
  }[]
) {
  const payload = {
    exported_at: new Date().toISOString(),
    count: rows.length,
    entries: rows.map(row => ({
      ...row,
      hash: row.hash || undefined,
      merkle_root: row.merkle_root || undefined,
      signature: row.signature || undefined
    }))
  }
  
  const blob = new Blob([JSON.stringify(payload, null, 2)], { 
    type: 'application/json' 
  })
  
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `policycortex-evidence-${Date.now()}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}