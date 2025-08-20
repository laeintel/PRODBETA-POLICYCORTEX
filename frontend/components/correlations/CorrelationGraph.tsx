'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  Filter,
  Info
} from 'lucide-react'

interface CorrelationGraphProps {
  correlations: any[]
  onNodeSelect: (nodeId: string | null) => void
  selectedNode: string | null
}

export default function CorrelationGraph({ correlations, onNodeSelect, selectedNode }: CorrelationGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [zoom, setZoom] = useState(1)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  // Simple force-directed graph visualization
  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Extract nodes from correlations
    const nodes = new Map()
    correlations.forEach(corr => {
      if (!nodes.has(corr.source.id)) {
        nodes.set(corr.source.id, {
          ...corr.source,
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height
        })
      }
      if (!nodes.has(corr.target.id)) {
        nodes.set(corr.target.id, {
          ...corr.target,
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height
        })
      }
    })

    // Draw edges
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'
    ctx.lineWidth = 2 * zoom
    correlations.forEach(corr => {
      const source = nodes.get(corr.source.id)
      const target = nodes.get(corr.target.id)
      if (source && target) {
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)
        ctx.stroke()

        // Draw correlation strength
        const midX = (source.x + target.x) / 2
        const midY = (source.y + target.y) / 2
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
        ctx.font = `${12 * zoom}px monospace`
        ctx.fillText(`${(corr.correlation_strength * 100).toFixed(0)}%`, midX, midY)
      }
    })

    // Draw nodes
    nodes.forEach((node, id) => {
      const isSelected = id === selectedNode
      const isHovered = id === hoveredNode
      
      // Node color based on risk
      const riskColor = node.risk > 0.7 ? '#ef4444' : node.risk > 0.4 ? '#f59e0b' : '#10b981'
      
      ctx.beginPath()
      ctx.arc(node.x, node.y, (20 + (isSelected ? 10 : 0)) * zoom, 0, Math.PI * 2)
      ctx.fillStyle = isSelected ? riskColor : `${riskColor}99`
      ctx.fill()
      
      if (isSelected || isHovered) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
      }
      
      // Node label
      ctx.fillStyle = 'white'
      ctx.font = `${11 * zoom}px monospace`
      ctx.textAlign = 'center'
      ctx.fillText(id, node.x, node.y + 35 * zoom)
    })
  }, [correlations, selectedNode, hoveredNode, zoom])

  return (
    <div className="relative bg-black/20 rounded-xl border border-white/10 overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
        <button
          onClick={() => setZoom(Math.min(zoom + 0.1, 2))}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <ZoomIn className="w-4 h-4 text-white" />
        </button>
        <button
          onClick={() => setZoom(Math.max(zoom - 0.1, 0.5))}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <ZoomOut className="w-4 h-4 text-white" />
        </button>
        <button
          onClick={() => setZoom(1)}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <Maximize2 className="w-4 h-4 text-white" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 z-10 bg-black/50 backdrop-blur rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-2">Risk Levels</div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span className="text-xs text-white">High (&gt;70%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full" />
            <span className="text-xs text-white">Medium (40-70%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="text-xs text-white">Low (&lt;40%)</span>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-[600px] cursor-move"
        onClick={(e) => {
          // Simple node selection (would need proper hit detection)
          onNodeSelect(null)
        }}
      />

      {/* Info Panel */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          className="absolute right-0 top-0 bottom-0 w-80 bg-black/80 backdrop-blur border-l border-white/10 p-4"
        >
          <h3 className="text-white font-semibold mb-3">Node Details</h3>
          <div className="space-y-2 text-sm">
            <div>
              <span className="text-gray-400">ID:</span>
              <span className="text-white ml-2">{selectedNode}</span>
            </div>
            <div>
              <span className="text-gray-400">Type:</span>
              <span className="text-white ml-2">Compute</span>
            </div>
            <div>
              <span className="text-gray-400">Risk Score:</span>
              <span className="text-yellow-400 ml-2">0.65</span>
            </div>
            <div>
              <span className="text-gray-400">Correlations:</span>
              <span className="text-white ml-2">{correlations.length}</span>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-white/10">
            <h4 className="text-white text-sm font-semibold mb-2">Related Correlations</h4>
            <div className="space-y-2">
              {correlations.slice(0, 3).map(corr => (
                <div key={corr.id} className="p-2 bg-white/5 rounded-lg">
                  <div className="text-xs text-gray-400">{corr.domain_pair.join(' → ')}</div>
                  <div className="text-xs text-white mt-1">{corr.description}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}