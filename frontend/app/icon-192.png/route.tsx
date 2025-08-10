import { ImageResponse } from 'next/og'

export const runtime = 'edge'

export function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #111827 0%, #1f2937 100%)',
          color: '#3b82f6',
          fontSize: 84,
          fontWeight: 800,
          letterSpacing: -2,
          borderRadius: 32,
        }}
      >
        PC
      </div>
    ),
    { width: 192, height: 192 }
  )
}


