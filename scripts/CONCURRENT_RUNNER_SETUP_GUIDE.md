# GitHub Self-Hosted Runner Concurrent Job Setup Guide

## Overview
Your Windows self-hosted GitHub Actions runner (`aeolitech-runner1`) currently processes one job at a time. This guide shows you how to configure it to run multiple jobs simultaneously, significantly improving your CI/CD pipeline performance.

## Current Status
- ✅ Windows self-hosted runner is working
- ❌ Only processes 1 job at a time
- 🎯 Goal: Enable 3-6 concurrent jobs

## Quick Setup (Recommended)

### Step 1: Check Your System Resources
```powershell
# Run this first to see how many concurrent jobs your system can handle
.\scripts\check-system-resources.ps1
```

### Step 2: Configure Concurrent Jobs
```powershell
# Replace with your actual runner directory path
.\scripts\configure-concurrent-runner.ps1 -RunnerDirectory "C:\actions-runner" -MaxConcurrentJobs 3
```

## Configuration Options

### Option 1: Single Runner with Concurrency (Easiest)
- Uses your existing `aeolitech-runner1`
- Allows 2-6 concurrent jobs on the same runner
- Minimal setup required
- **Recommended for most users**

**Command:**
```powershell
.\scripts\configure-concurrent-runner.ps1 -RunnerDirectory "C:\actions-runner" -MaxConcurrentJobs 3
```

### Option 2: Multiple Runner Instances (Advanced)
- Creates 3 separate runners: `aeolitech-runner1`, `aeolitech-runner2`, `aeolitech-runner3`
- Each runner processes jobs independently
- Maximum parallelism
- Requires GitHub token

**Command:**
```powershell
.\scripts\setup-multiple-runners.ps1 -GitHubToken "your-github-token" -RunnerCount 3
```

## What Files Were Created

1. **check-system-resources.ps1**
   - Analyzes your PC's CPU, RAM, and disk space
   - Recommends optimal number of concurrent jobs
   - Shows current runner service status

2. **configure-concurrent-runner.ps1**
   - Reconfigures your existing runner for concurrent jobs
   - Stops, updates, and restarts the runner service
   - Safest and easiest option

3. **setup-multiple-runners.ps1**
   - Creates multiple independent runner instances
   - Downloads and configures additional runners
   - More complex but provides maximum parallelism

4. **optimize-workflow-concurrency.yml**
   - Example workflow configuration
   - Shows how to structure jobs for parallel execution
   - Use as reference for updating your pipelines

## Expected Results After Setup

### Before (Current State):
```
Job 1: Running on aeolitech-runner1
Job 2: Waiting in queue
Job 3: Waiting in queue
```

### After (With Concurrency):
```
Job 1: Running on aeolitech-runner1 (slot 1)
Job 2: Running on aeolitech-runner1 (slot 2)  
Job 3: Running on aeolitech-runner1 (slot 3)
```

## Performance Benefits

- **3-6x faster pipeline execution**
- **Parallel service testing** (all 6 services test simultaneously)
- **Concurrent frontend/backend jobs**
- **Better resource utilization** of your Windows PC

## System Requirements

### Minimum for 3 Concurrent Jobs:
- 4+ CPU cores
- 8+ GB RAM
- 20+ GB free disk space

### Recommended for 6 Concurrent Jobs:
- 8+ CPU cores  
- 16+ GB RAM
- 50+ GB free disk space

## Workflow Updates

Your existing `.github/workflows/ci-cd-pipeline.yml` will automatically benefit from concurrent execution. No changes required, but you can optimize further by:

1. Using `strategy.matrix.max-parallel` settings
2. Structuring independent jobs to run in parallel
3. Avoiding unnecessary job dependencies

## Troubleshooting

### If Runner Service Fails to Start:
1. Run PowerShell as Administrator
2. Check Windows Event Viewer for service errors
3. Verify runner directory path is correct
4. Ensure GitHub token has proper permissions

### If Jobs Still Queue:
1. Verify service is running: `Get-Service "actions.runner.*"`
2. Check GitHub repository settings > Actions > Runners
3. Monitor system resources during job execution
4. Consider reducing concurrent job count if system struggles

### If System Becomes Slow:
1. Reduce `-MaxConcurrentJobs` parameter
2. Add more RAM if possible
3. Monitor CPU usage in Task Manager
4. Consider upgrading hardware

## Next Steps

1. **Run system check** to understand your PC's capabilities
2. **Choose configuration option** (Option 1 recommended)
3. **Execute setup script** with Administrator privileges
4. **Test with multiple pipeline runs** to verify concurrency
5. **Monitor system performance** and adjust as needed

## Verification Commands

```powershell
# Check if runner service is running with concurrency
Get-Service "actions.runner.*" | Format-Table Name, Status

# Monitor runner processes during job execution  
Get-Process | Where-Object {$_.ProcessName -like "*runner*"}

# Check system resource usage
Get-Counter "\Processor(_Total)\% Processor Time"
```

## Support

If you encounter issues:
1. Check the PowerShell execution output for errors
2. Verify your GitHub repository has the runner registered
3. Ensure Windows Defender/Antivirus isn't blocking runner processes
4. Check available system resources before increasing concurrency

---

**Ready to get started?** Run `.\scripts\check-system-resources.ps1` first to see what your system can handle!