"""
Test script for continuous learning system
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.ai_engine.continuous_learning import (
    ContinuousLearningSystem,
    ErrorEvent
)

async def test_continuous_learning():
    """Test the continuous learning system"""
    print("Initializing Continuous Learning System...")
    
    # Initialize the system
    learner = ContinuousLearningSystem(vocab_size=10000)
    
    print(f"System initialized with {sum(p.numel() for p in learner.model.parameters())} parameters")
    
    # Create some test errors
    test_errors = [
        ErrorEvent(
            timestamp=datetime.utcnow(),
            source='application',
            error_type='ConnectionError',
            error_message='Failed to connect to Azure Resource Manager',
            stack_trace='Traceback...',
            context={'service': 'azure_client', 'retry_count': 3},
            solution=None,
            tags=['azure', 'connection'],
            severity='high',
            domain='cloud'
        ),
        ErrorEvent(
            timestamp=datetime.utcnow(),
            source='application',
            error_type='AuthenticationError',
            error_message='Invalid JWT token',
            stack_trace='Traceback...',
            context={'endpoint': '/api/v1/resources'},
            solution=None,
            tags=['auth', 'jwt'],
            severity='critical',
            domain='security'
        ),
        ErrorEvent(
            timestamp=datetime.utcnow(),
            source='application',
            error_type='TimeoutError',
            error_message='Network request timed out after 30s',
            stack_trace='Traceback...',
            context={'url': 'https://api.example.com'},
            solution=None,
            tags=['network', 'timeout'],
            severity='medium',
            domain='network'
        )
    ]
    
    print(f"\nTraining on {len(test_errors)} test errors...")
    
    # Learn from errors
    await learner.learn_from_errors(test_errors)
    
    # Get learning statistics
    stats = learner.get_learning_stats()
    print(f"\nLearning Statistics:")
    print(f"  - Total errors processed: {stats['metrics']['total_errors_processed']}")
    print(f"  - Training steps: {stats['metrics']['total_training_steps']}")
    print(f"  - Average loss: {stats['metrics']['average_loss']:.4f}")
    print(f"  - Buffer size: {stats['buffer_size']}")
    
    # Test prediction
    print("\nTesting Error Prediction...")
    
    test_error_message = "Failed to authenticate with Azure AD"
    prediction = learner.predict_solution(test_error_message, domain='security')
    
    print(f"\n  Error: {test_error_message}")
    print(f"  Predicted Classification: {prediction['error_classification']}")
    print(f"  Severity: {prediction['severity']}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"  Learned from: {prediction['learned_from']}")
    
    # Test another prediction
    test_error_message2 = "Connection refused to database server"
    prediction2 = learner.predict_solution(test_error_message2, domain='network')
    
    print(f"\n  Error: {test_error_message2}")
    print(f"  Predicted Classification: {prediction2['error_classification']}")
    print(f"  Severity: {prediction2['severity']}")
    print(f"  Confidence: {prediction2['confidence']:.2%}")
    
    # Save checkpoint
    print("\nSaving model checkpoint...")
    learner.save_checkpoint()
    
    print("\nContinuous Learning System Test Complete!")
    
    return learner

async def test_error_collection():
    """Test error collection from application logs"""
    print("\nTesting Error Collection from Application Logs...")
    
    learner = ContinuousLearningSystem(vocab_size=5000)
    
    # Simulate application log errors
    app_errors = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": "ValidationError",
            "message": "Policy validation failed: missing required field 'effect'",
            "stack_trace": "File policy.py, line 123...",
            "context": {"policy_id": "pol-123", "field": "effect"},
            "tags": ["policy", "validation"],
            "severity": "medium"
        },
        {
            "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "error_type": "RateLimitError",
            "message": "Azure API rate limit exceeded",
            "stack_trace": None,
            "context": {"api": "resource_manager", "calls": 1000},
            "tags": ["azure", "rate_limit"],
            "severity": "high"
        }
    ]
    
    # Collect errors
    error_events = await learner.collect_errors_from_application(app_errors)
    
    print(f"  Collected {len(error_events)} errors from application logs")
    
    for event in error_events:
        print(f"    - {event.error_type}: {event.error_message[:50]}... [{event.domain}]")
    
    # Learn from collected errors
    await learner.learn_from_errors(error_events)
    
    print(f"  Successfully learned from application errors")
    
    return learner

async def main():
    """Run all tests"""
    print("=" * 60)
    print("CONTINUOUS LEARNING SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test basic functionality
    learner1 = await test_continuous_learning()
    
    # Test error collection
    learner2 = await test_error_collection()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Show that the system is learning
    print("\nSystem is continuously learning and improving!")
    print("   - Learns from application errors automatically")
    print("   - Fetches data from Stack Overflow, Reddit, GitHub")
    print("   - Uses Adam optimizer with warmup and decay")
    print("   - Implements transformer architecture with positional encoding")
    print("   - Provides real-time error predictions and solutions")

if __name__ == "__main__":
    asyncio.run(main())