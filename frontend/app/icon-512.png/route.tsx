import { ImageResponse } from 'next/og'

export const runtime = 'edge'
export const contentType = 'image/png'
export const size = { width: 512, height: 512 }

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
          fontSize: 180,
          fontWeight: 800,
          letterSpacing: -6,
          borderRadius: 64,
        }}
      >
        PC
      </div>
    ),
    { width: size.width, height: size.height }
  )
}


