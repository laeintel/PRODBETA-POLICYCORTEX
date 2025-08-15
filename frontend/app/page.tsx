'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  Shield, Brain, Zap, Globe, Lock, BarChart, Users, DollarSign, 
  Server, Network, Activity, Sparkles, LogIn, ChevronRight,
  CheckCircle, ArrowRight, Play, Star, Award, TrendingUp,
  Cloud, Eye, Cpu, Database, Gauge, ShieldCheck
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated, login, user, loading } = useAuth()
  const [currentFeature, setCurrentFeature] = useState(0)

  const handleLogin = async () => {
    try {
      await login()
      router.push('/dashboard')
    } catch (error) {
      console.error('Login failed:', error)
    }
  }

  const handleGetStarted = () => {
    if (isAuthenticated) {
      router.push('/dashboard')
    } else {
      handleLogin()
    }
  }

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Governance",
      description: "Advanced machine learning for intelligent policy automation",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: Shield,
      title: "Enterprise Security",
      description: "Post-quantum cryptography with zero-trust architecture",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: TrendingUp,
      title: "Cost Optimization",
      description: "Real-time cost analysis with predictive recommendations",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: Network,
      title: "Network Intelligence",
      description: "Deep network analysis with threat detection",
      color: "from-orange-500 to-red-500"
    }
  ]

  const stats = [
    { label: "Azure Resources Managed", value: "10K+", icon: Cloud },
    { label: "Cost Savings Achieved", value: "40%", icon: DollarSign },
    { label: "Security Threats Blocked", value: "99.9%", icon: ShieldCheck },
    { label: "Response Time", value: "<100ms", icon: Gauge }
  ]

  const patents = [
    "Cross-Domain Governance Correlation Engine",
    "Conversational Governance Intelligence System", 
    "Unified AI-Driven Cloud Governance Platform",
    "Predictive Policy Compliance Engine"
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length)
    }, 4000)
    return () => clearInterval(interval)
  }, [features.length])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/20 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500/20 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-500/10 rounded-full mix-blend-multiply filter blur-2xl animate-pulse delay-2000"></div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 container mx-auto px-6 py-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Shield className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white">PolicyCortex</span>
            <span className="text-sm bg-purple-600 text-white px-2 py-1 rounded-full">v2</span>
          </div>
          
          <div className="flex items-center space-x-6">
            {!isAuthenticated ? (
              <button 
                onClick={handleLogin}
                disabled={loading}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-all duration-200 flex items-center gap-2 disabled:opacity-50 hover:scale-105"
              >
                <LogIn className="w-4 h-4" />
                {loading ? 'Signing in...' : 'Login with Azure AD'}
              </button>
            ) : (
              <div className="flex items-center gap-4">
                <div className="text-green-400 text-sm flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  {user?.name || user?.username}
                </div>
                <button 
                  onClick={() => router.push('/dashboard')}
                  className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-200 flex items-center gap-2 hover:scale-105"
                >
                  Dashboard
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <div className="relative z-10 container mx-auto px-6">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center py-20"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="mb-8"
          >
            <h1 className="text-7xl lg:text-8xl font-bold text-white mb-6 leading-tight">
              The Future of
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent block">
                Cloud Governance
              </span>
            </h1>
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="text-xl lg:text-2xl text-gray-300 mb-4 max-w-4xl mx-auto"
          >
            Enterprise-grade AI platform with <span className="text-purple-400 font-semibold">4 patented technologies</span> for 
            complete Azure governance, security, and cost optimization.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.6 }}
            className="flex flex-wrap justify-center gap-3 mb-12 text-sm"
          >
            {['RBAC Management', 'Cost Optimization', 'Policy Automation', 'Network Security', 'AI Training', 'Real-time Analytics'].map((item, index) => (
              <span key={index} className="px-4 py-2 bg-white/10 rounded-full text-gray-300 backdrop-blur-sm border border-white/20">
                {item}
              </span>
            ))}
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="flex flex-col sm:flex-row gap-6 justify-center items-center"
          >
            <button 
              onClick={handleGetStarted}
              disabled={loading}
              className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 flex items-center gap-3 disabled:opacity-50 transform hover:scale-105 shadow-2xl hover:shadow-purple-500/25"
            >
              <Sparkles className="w-5 h-5" />
              {isAuthenticated ? 'Enter Dashboard' : 'Start Free Trial'}
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
            
            <button 
              onClick={() => router.push('/features')}
              className="group px-8 py-4 bg-white/10 text-white rounded-xl font-semibold backdrop-blur-md hover:bg-white/20 transition-all duration-300 border border-white/20 flex items-center gap-3 hover:scale-105"
            >
              <Play className="w-5 h-5" />
              Watch Demo
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </motion.div>
        </motion.div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.8 }}
          className="grid grid-cols-2 lg:grid-cols-4 gap-8 mb-20"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.2 + index * 0.1, duration: 0.5 }}
              className="text-center p-6 bg-white/5 rounded-2xl backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all duration-300"
            >
              <stat.icon className="w-8 h-8 text-purple-400 mx-auto mb-3" />
              <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
              <div className="text-gray-400 text-sm">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Features Carousel */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-12">
            Powered by <span className="text-purple-400">Advanced AI</span>
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              {features.map((feature, index) => (
                <motion.div
                  key={index}
                  className={`p-6 rounded-2xl border transition-all duration-500 cursor-pointer ${
                    index === currentFeature 
                      ? 'bg-white/10 border-purple-500/50 scale-105' 
                      : 'bg-white/5 border-white/10 hover:bg-white/8'
                  }`}
                  onClick={() => setCurrentFeature(index)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-r ${feature.color}`}>
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                      <p className="text-gray-400">{feature.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
            
            <div className="relative">
              <motion.div
                key={currentFeature}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
                className="bg-gradient-to-br from-white/10 to-white/5 p-8 rounded-3xl backdrop-blur-sm border border-white/20"
              >
                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${features[currentFeature].color} flex items-center justify-center mb-6`}>
                  <features[currentFeature].icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">{features[currentFeature].title}</h3>
                <p className="text-gray-300 text-lg leading-relaxed">{features[currentFeature].description}</p>
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* Patents Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.6, duration: 0.8 }}
          className="text-center mb-20"
        >
          <div className="flex items-center justify-center gap-2 mb-6">
            <Award className="w-6 h-6 text-yellow-400" />
            <h2 className="text-3xl font-bold text-white">4 Patented Technologies</h2>
            <Award className="w-6 h-6 text-yellow-400" />
          </div>
          
          <div className="grid md:grid-cols-2 gap-4 max-w-4xl mx-auto">
            {patents.map((patent, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.8 + index * 0.1, duration: 0.5 }}
                className="p-4 bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-xl border border-purple-500/30 text-left"
              >
                <div className="flex items-start gap-3">
                  <Star className="w-5 h-5 text-yellow-400 mt-1 flex-shrink-0" />
                  <span className="text-gray-300 font-medium">{patent}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2, duration: 0.8 }}
          className="text-center py-20"
        >
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            Ready to Transform Your 
            <span className="text-purple-400"> Cloud Governance?</span>
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Join leading enterprises using PolicyCortex for intelligent, automated cloud management.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <button 
              onClick={handleGetStarted}
              disabled={loading}
              className="group px-10 py-5 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl font-bold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 flex items-center justify-center gap-3 disabled:opacity-50 transform hover:scale-105 shadow-2xl text-lg"
            >
              <Sparkles className="w-6 h-6" />
              {isAuthenticated ? 'Access Dashboard' : 'Start Your Journey'}
              <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          <div className="mt-12 flex flex-wrap justify-center gap-8 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span>No Credit Card Required</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span>Enterprise-Grade Security</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span>24/7 Support</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}