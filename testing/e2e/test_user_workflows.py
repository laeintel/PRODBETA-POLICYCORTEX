"""
End-to-End User Workflow Tests
Tests complete user journeys through the PolicyCortex platform
"""

import pytest
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import asyncio
from typing import Dict, Any
import os
import json

class TestUserWorkflows:
    """End-to-end tests for complete user workflows"""
    
    @pytest.fixture(scope="class")
    async def browser_setup(self, e2e_test_environment: Dict[str, Any]):
        """Set up browser for testing"""
        playwright = await async_playwright().start()
        
        browser = await playwright.chromium.launch(
            headless=e2e_test_environment["headless"],
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        
        yield browser, playwright
        
        await browser.close()
        await playwright.stop()
    
    @pytest.fixture
    async def authenticated_page(
        self,
        browser_setup,
        e2e_test_environment: Dict[str, Any]
    ) -> Page:
        """Create authenticated browser page"""
        browser, _ = browser_setup
        
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir="testing/reports/videos" if os.getenv("RECORD_VIDEOS") == "true" else None
        )
        
        page = await context.new_page()
        
        # Navigate to application
        await page.goto(e2e_test_environment["frontend_url"])
        
        # Perform authentication
        await self._authenticate_user(page, e2e_test_environment)
        
        yield page
        
        await context.close()
    
    async def _authenticate_user(self, page: Page, env: Dict[str, Any]):
        """Authenticate user in the application"""
        try:
            # Wait for login page to load
            await page.wait_for_selector("[data-testid='login-form']", timeout=10000)
            
            # Fill login form
            await page.fill("[data-testid='email-input']", env["test_user_email"])
            await page.fill("[data-testid='password-input']", env["test_user_password"])
            
            # Click login button
            await page.click("[data-testid='login-button']")
            
            # Wait for successful authentication
            await page.wait_for_selector("[data-testid='dashboard']", timeout=15000)
            
        except Exception as e:
            # Take screenshot for debugging
            await page.screenshot(path="testing/reports/screenshots/auth_failure.png")
            raise e
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_new_user_onboarding_workflow(
        self,
        authenticated_page: Page,
        e2e_test_environment: Dict[str, Any]
    ):
        """Test complete new user onboarding workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to onboarding
        await page.click("[data-testid='start-onboarding-button']")
        await page.wait_for_selector("[data-testid='onboarding-wizard']")
        
        # Step 2: Organization Information
        await page.fill("[data-testid='organization-name']", "Test Organization E2E")
        await page.select_option("[data-testid='organization-type']", "enterprise")
        await page.select_option("[data-testid='industry']", "technology")
        await page.fill("[data-testid='employee-count']", "500")
        await page.click("[data-testid='next-button']")
        
        # Step 3: Azure Connection
        await page.wait_for_selector("[data-testid='azure-connection-form']")
        await page.fill("[data-testid='subscription-id']", "test-subscription-id")
        await page.fill("[data-testid='tenant-id']", "test-tenant-id")
        await page.fill("[data-testid='client-id']", "test-client-id")
        await page.fill("[data-testid='client-secret']", "test-client-secret")
        
        # Test connection
        await page.click("[data-testid='test-connection-button']")
        await page.wait_for_selector("[data-testid='connection-success']", timeout=10000)
        
        await page.click("[data-testid='next-button']")
        
        # Step 4: Feature Selection
        await page.wait_for_selector("[data-testid='feature-selection']")
        await page.check("[data-testid='feature-compliance']")
        await page.check("[data-testid='feature-analytics']")
        await page.check("[data-testid='feature-notifications']")
        await page.click("[data-testid='next-button']")
        
        # Step 5: Initial Scan
        await page.wait_for_selector("[data-testid='initial-scan']")
        await page.click("[data-testid='start-scan-button']")
        
        # Wait for scan to complete
        await page.wait_for_selector("[data-testid='scan-complete']", timeout=60000)
        
        # Verify scan results
        scan_results = await page.text_content("[data-testid='scan-summary']")
        assert "resources discovered" in scan_results.lower()
        
        await page.click("[data-testid='next-button']")
        
        # Step 6: User Invitations
        await page.wait_for_selector("[data-testid='user-invitations']")
        await page.fill("[data-testid='invite-email-0']", "colleague1@test.com")
        await page.select_option("[data-testid='invite-role-0']", "viewer")
        
        await page.click("[data-testid='add-another-invite']")
        await page.fill("[data-testid='invite-email-1']", "colleague2@test.com")
        await page.select_option("[data-testid='invite-role-1']", "admin")
        
        await page.click("[data-testid='send-invitations']")
        await page.click("[data-testid='next-button']")
        
        # Step 7: Configuration
        await page.wait_for_selector("[data-testid='configuration']")
        await page.select_option("[data-testid='notification-frequency']", "daily")
        await page.check("[data-testid='enable-auto-remediation']")
        await page.click("[data-testid='next-button']")
        
        # Step 8: Completion
        await page.wait_for_selector("[data-testid='onboarding-complete']", timeout=30000)
        
        completion_message = await page.text_content("[data-testid='completion-message']")
        assert "welcome to policycortex" in completion_message.lower()
        
        await page.click("[data-testid='go-to-dashboard']")
        
        # Verify dashboard is loaded with onboarded data
        await page.wait_for_selector("[data-testid='dashboard']")
        await page.wait_for_selector("[data-testid='compliance-overview']")
        
        # Take success screenshot
        await page.screenshot(path="testing/reports/screenshots/onboarding_success.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_policy_creation_and_compliance_monitoring(
        self,
        authenticated_page: Page
    ):
        """Test policy creation and compliance monitoring workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to Policies
        await page.click("[data-testid='nav-policies']")
        await page.wait_for_selector("[data-testid='policies-page']")
        
        # Step 2: Create New Policy
        await page.click("[data-testid='create-policy-button']")
        await page.wait_for_selector("[data-testid='policy-wizard']")
        
        # Basic Information
        await page.fill("[data-testid='policy-name']", "E2E Test Storage Security Policy")
        await page.fill("[data-testid='policy-description']", "Ensure all storage accounts use HTTPS and encryption")
        await page.select_option("[data-testid='policy-category']", "security")
        await page.select_option("[data-testid='policy-severity']", "high")
        
        await page.click("[data-testid='next-step']")
        
        # Rule Builder
        await page.wait_for_selector("[data-testid='rule-builder']")
        
        # First rule: HTTPS enforcement
        await page.select_option("[data-testid='resource-type']", "Microsoft.Storage/storageAccounts")
        await page.select_option("[data-testid='property']", "properties.supportsHttpsTrafficOnly")
        await page.select_option("[data-testid='operator']", "equals")
        await page.fill("[data-testid='value']", "true")
        await page.fill("[data-testid='rule-message']", "Storage accounts must use HTTPS only")
        
        await page.click("[data-testid='add-rule']")
        
        # Second rule: Encryption at rest
        await page.select_option("[data-testid='resource-type']", "Microsoft.Storage/storageAccounts")
        await page.select_option("[data-testid='property']", "properties.encryption.services.blob.enabled")
        await page.select_option("[data-testid='operator']", "equals")
        await page.fill("[data-testid='value']", "true")
        await page.fill("[data-testid='rule-message']", "Storage accounts must have blob encryption enabled")
        
        await page.click("[data-testid='next-step']")
        
        # Review and Create
        await page.wait_for_selector("[data-testid='policy-review']")
        
        # Verify policy details in review
        policy_name = await page.text_content("[data-testid='review-policy-name']")
        assert "E2E Test Storage Security Policy" in policy_name
        
        rules_count = await page.text_content("[data-testid='review-rules-count']")
        assert "2" in rules_count
        
        await page.click("[data-testid='create-policy']")
        
        # Wait for creation success
        await page.wait_for_selector("[data-testid='policy-created-success']")
        
        # Step 3: Run Compliance Analysis
        await page.click("[data-testid='run-analysis-button']")
        await page.wait_for_selector("[data-testid='analysis-progress']")
        
        # Wait for analysis to complete
        await page.wait_for_selector("[data-testid='analysis-complete']", timeout=60000)
        
        # Step 4: View Compliance Results
        await page.click("[data-testid='view-results-button']")
        await page.wait_for_selector("[data-testid='compliance-results']")
        
        # Verify results are displayed
        results_summary = await page.text_content("[data-testid='results-summary']")
        assert "compliance" in results_summary.lower()
        
        # Check for resource details
        await page.wait_for_selector("[data-testid='resource-list']")
        
        # Click on first non-compliant resource (if any)
        non_compliant = await page.query_selector("[data-testid='resource-non-compliant']")
        if non_compliant:
            await non_compliant.click()
            
            # Verify resource details modal
            await page.wait_for_selector("[data-testid='resource-details-modal']")
            
            resource_details = await page.text_content("[data-testid='resource-details']")
            assert "storage" in resource_details.lower()
            
            await page.click("[data-testid='close-modal']")
        
        # Step 5: Generate Report
        await page.click("[data-testid='generate-report-button']")
        await page.wait_for_selector("[data-testid='report-options']")
        
        await page.select_option("[data-testid='report-format']", "pdf")
        await page.check("[data-testid='include-details']")
        await page.check("[data-testid='include-recommendations']")
        
        await page.click("[data-testid='generate-report']")
        
        # Wait for report generation
        await page.wait_for_selector("[data-testid='report-ready']", timeout=30000)
        
        # Take screenshot of results
        await page.screenshot(path="testing/reports/screenshots/compliance_results.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_conversational_ai_workflow(
        self,
        authenticated_page: Page
    ):
        """Test conversational AI interface workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to AI Assistant
        await page.click("[data-testid='nav-ai-assistant']")
        await page.wait_for_selector("[data-testid='ai-chat-interface']")
        
        # Step 2: Ask about compliance status
        question1 = "What is our current compliance status?"
        await page.fill("[data-testid='chat-input']", question1)
        await page.click("[data-testid='send-message']")
        
        # Wait for response
        await page.wait_for_selector("[data-testid='ai-response']", timeout=15000)
        
        response1 = await page.text_content("[data-testid='ai-response']:last-child")
        assert any(keyword in response1.lower() for keyword in ['compliance', 'policy', 'status'])
        
        # Step 3: Ask for specific policy information
        question2 = "Show me all high severity security policies"
        await page.fill("[data-testid='chat-input']", question2)
        await page.click("[data-testid='send-message']")
        
        await page.wait_for_selector("[data-testid='ai-response']:nth-child(4)", timeout=15000)
        
        # Verify response includes policy information
        response2 = await page.text_content("[data-testid='ai-response']:nth-child(4)")
        assert any(keyword in response2.lower() for keyword in ['security', 'policy', 'high'])
        
        # Step 4: Request recommendations
        question3 = "What are your recommendations to improve our security posture?"
        await page.fill("[data-testid='chat-input']", question3)
        await page.click("[data-testid='send-message']")
        
        await page.wait_for_selector("[data-testid='ai-response']:nth-child(6)", timeout=15000)
        
        response3 = await page.text_content("[data-testid='ai-response']:nth-child(6)")
        assert any(keyword in response3.lower() for keyword in ['recommend', 'improve', 'security'])
        
        # Step 5: Test conversation context
        question4 = "Can you explain the first recommendation in more detail?"
        await page.fill("[data-testid='chat-input']", question4)
        await page.click("[data-testid='send-message']")
        
        await page.wait_for_selector("[data-testid='ai-response']:nth-child(8)", timeout=15000)
        
        response4 = await page.text_content("[data-testid='ai-response']:nth-child(8)")
        assert len(response4) > 50  # Should be a detailed explanation
        
        # Step 6: Test action triggers
        question5 = "Create a new policy for database encryption"
        await page.fill("[data-testid='chat-input']", question5)
        await page.click("[data-testid='send-message']")
        
        await page.wait_for_selector("[data-testid='ai-response']:nth-child(10)", timeout=15000)
        
        # Look for action suggestions or buttons
        action_buttons = await page.query_selector_all("[data-testid='suggested-action']")
        assert len(action_buttons) > 0
        
        # Click on suggested action if available
        if action_buttons:
            await action_buttons[0].click()
            
            # Should navigate to policy creation or show policy template
            await page.wait_for_selector("[data-testid='policy-template']", timeout=10000)
        
        # Take screenshot of conversation
        await page.screenshot(path="testing/reports/screenshots/ai_conversation.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_analytics_and_insights_workflow(
        self,
        authenticated_page: Page
    ):
        """Test analytics dashboard and insights workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to Analytics
        await page.click("[data-testid='nav-analytics']")
        await page.wait_for_selector("[data-testid='analytics-dashboard']")
        
        # Step 2: Verify dashboard components load
        await page.wait_for_selector("[data-testid='compliance-trends-chart']")
        await page.wait_for_selector("[data-testid='cost-analysis-chart']")
        await page.wait_for_selector("[data-testid='security-score-gauge']")
        await page.wait_for_selector("[data-testid='top-risks-list']")
        
        # Step 3: Interact with time range selector
        await page.click("[data-testid='time-range-selector']")
        await page.click("[data-testid='time-range-30d']")
        
        # Wait for charts to update
        await page.wait_for_timeout(2000)
        
        # Step 4: Generate custom insights
        await page.click("[data-testid='generate-insights-button']")
        await page.wait_for_selector("[data-testid='insight-options']")
        
        await page.check("[data-testid='insight-compliance-trends']")
        await page.check("[data-testid='insight-cost-optimization']")
        await page.check("[data-testid='insight-security-recommendations']")
        
        await page.click("[data-testid='generate-insights']")
        
        # Wait for insights generation
        await page.wait_for_selector("[data-testid='insights-ready']", timeout=30000)
        
        # Step 5: View detailed insights
        await page.click("[data-testid='view-insights']")
        await page.wait_for_selector("[data-testid='insights-detail-view']")
        
        # Verify insights content
        insights = await page.query_selector_all("[data-testid='insight-item']")
        assert len(insights) >= 3
        
        # Click on first insight
        await insights[0].click()
        await page.wait_for_selector("[data-testid='insight-detail-modal']")
        
        # Verify insight details
        insight_content = await page.text_content("[data-testid='insight-content']")
        assert len(insight_content) > 100  # Should have substantial content
        
        # Step 6: Export insights
        await page.click("[data-testid='export-insights-button']")
        await page.wait_for_selector("[data-testid='export-options']")
        
        await page.select_option("[data-testid='export-format']", "pdf")
        await page.click("[data-testid='export-insights']")
        
        # Wait for export completion
        await page.wait_for_selector("[data-testid='export-complete']", timeout=20000)
        
        await page.click("[data-testid='close-modal']")
        
        # Step 7: Test predictive analytics
        await page.click("[data-testid='predictive-analytics-tab']")
        await page.wait_for_selector("[data-testid='predictive-models']")
        
        # View compliance predictions
        await page.click("[data-testid='compliance-predictions']")
        await page.wait_for_selector("[data-testid='prediction-chart']")
        
        prediction_summary = await page.text_content("[data-testid='prediction-summary']")
        assert any(keyword in prediction_summary.lower() for keyword in ['predict', 'forecast', 'trend'])
        
        # Take screenshot of analytics
        await page.screenshot(path="testing/reports/screenshots/analytics_dashboard.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_notification_and_alert_workflow(
        self,
        authenticated_page: Page
    ):
        """Test notification and alert management workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to Notifications
        await page.click("[data-testid='nav-notifications']")
        await page.wait_for_selector("[data-testid='notifications-center']")
        
        # Step 2: View current alerts
        await page.wait_for_selector("[data-testid='alerts-list']")
        
        alerts = await page.query_selector_all("[data-testid='alert-item']")
        
        if alerts:
            # Click on first alert
            await alerts[0].click()
            await page.wait_for_selector("[data-testid='alert-details-modal']")
            
            # Acknowledge alert
            await page.click("[data-testid='acknowledge-alert']")
            await page.fill("[data-testid='acknowledgment-notes']", "Investigating this issue")
            await page.click("[data-testid='confirm-acknowledgment']")
            
            # Verify alert status changed
            await page.wait_for_selector("[data-testid='alert-acknowledged']")
            await page.click("[data-testid='close-modal']")
        
        # Step 3: Configure notification preferences
        await page.click("[data-testid='notification-settings']")
        await page.wait_for_selector("[data-testid='notification-preferences']")
        
        # Email preferences
        await page.check("[data-testid='email-notifications']")
        await page.select_option("[data-testid='email-frequency']", "immediate")
        await page.check("[data-testid='email-critical-alerts']")
        await page.check("[data-testid='email-compliance-reports']")
        
        # SMS preferences
        await page.check("[data-testid='sms-notifications']")
        await page.fill("[data-testid='sms-phone-number']", "+1234567890")
        await page.check("[data-testid='sms-critical-only']")
        
        # Slack integration
        await page.check("[data-testid='slack-notifications']")
        await page.fill("[data-testid='slack-webhook-url']", "https://hooks.slack.com/test")
        
        await page.click("[data-testid='save-preferences']")
        await page.wait_for_selector("[data-testid='preferences-saved']")
        
        # Step 4: Create custom alert rule
        await page.click("[data-testid='create-alert-rule']")
        await page.wait_for_selector("[data-testid='alert-rule-wizard']")
        
        await page.fill("[data-testid='rule-name']", "E2E Test Critical Storage Alert")
        await page.select_option("[data-testid='rule-type']", "compliance_violation")
        await page.select_option("[data-testid='rule-severity']", "critical")
        
        # Conditions
        await page.select_option("[data-testid='condition-resource-type']", "storage_account")
        await page.select_option("[data-testid='condition-property']", "https_only")
        await page.select_option("[data-testid='condition-operator']", "equals")
        await page.fill("[data-testid='condition-value']", "false")
        
        # Escalation
        await page.check("[data-testid='enable-escalation']")
        await page.fill("[data-testid='escalation-delay']", "15")  # 15 minutes
        
        await page.click("[data-testid='create-alert-rule']")
        await page.wait_for_selector("[data-testid='alert-rule-created']")
        
        # Step 5: Test alert simulation
        await page.click("[data-testid='test-alert-simulation']")
        await page.wait_for_selector("[data-testid='simulation-options']")
        
        await page.select_option("[data-testid='simulation-type']", "compliance_violation")
        await page.fill("[data-testid='simulation-resource']", "test-storage-account")
        
        await page.click("[data-testid='run-simulation']")
        
        # Wait for simulated alert
        await page.wait_for_selector("[data-testid='simulation-alert-created']", timeout=10000)
        
        # Verify alert appears in list
        await page.wait_for_selector("[data-testid='alerts-list']")
        
        new_alerts = await page.query_selector_all("[data-testid='alert-item']")
        assert len(new_alerts) > len(alerts)  # New alert should be added
        
        # Take screenshot of notifications
        await page.screenshot(path="testing/reports/screenshots/notifications_center.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_user_management_workflow(
        self,
        authenticated_page: Page
    ):
        """Test user management and permissions workflow"""
        
        page = authenticated_page
        
        # Step 1: Navigate to User Management (admin only)
        await page.click("[data-testid='nav-admin']")
        await page.wait_for_selector("[data-testid='admin-panel']")
        
        await page.click("[data-testid='user-management-tab']")
        await page.wait_for_selector("[data-testid='user-list']")
        
        # Step 2: Invite new user
        await page.click("[data-testid='invite-user-button']")
        await page.wait_for_selector("[data-testid='invite-user-modal']")
        
        await page.fill("[data-testid='invite-email']", "newuser@test.com")
        await page.fill("[data-testid='invite-first-name']", "New")
        await page.fill("[data-testid='invite-last-name']", "User")
        await page.select_option("[data-testid='invite-role']", "analyst")
        
        # Set permissions
        await page.check("[data-testid='permission-view-policies']")
        await page.check("[data-testid='permission-create-reports']")
        await page.uncheck("[data-testid='permission-admin-access']")
        
        await page.click("[data-testid='send-invitation']")
        await page.wait_for_selector("[data-testid='invitation-sent']")
        
        # Step 3: Manage existing user
        users = await page.query_selector_all("[data-testid='user-row']")
        if users:
            # Click on first user
            await users[0].click()
            await page.wait_for_selector("[data-testid='user-details-modal']")
            
            # Update user role
            await page.select_option("[data-testid='user-role']", "senior_analyst")
            
            # Update permissions
            await page.check("[data-testid='permission-create-policies']")
            
            await page.click("[data-testid='save-user-changes']")
            await page.wait_for_selector("[data-testid='user-updated']")
            
            await page.click("[data-testid='close-modal']")
        
        # Step 4: Create custom role
        await page.click("[data-testid='roles-tab']")
        await page.wait_for_selector("[data-testid='roles-list']")
        
        await page.click("[data-testid='create-role-button']")
        await page.wait_for_selector("[data-testid='create-role-modal']")
        
        await page.fill("[data-testid='role-name']", "E2E Test Custom Role")
        await page.fill("[data-testid='role-description']", "Custom role for E2E testing")
        
        # Set permissions for custom role
        await page.check("[data-testid='permission-view-dashboard']")
        await page.check("[data-testid='permission-view-policies']")
        await page.check("[data-testid='permission-create-reports']")
        await page.check("[data-testid='permission-manage-notifications']")
        
        await page.click("[data-testid='create-role']")
        await page.wait_for_selector("[data-testid='role-created']")
        
        # Step 5: View audit logs
        await page.click("[data-testid='audit-logs-tab']")
        await page.wait_for_selector("[data-testid='audit-logs-list']")
        
        # Filter audit logs
        await page.select_option("[data-testid='audit-event-type']", "user_management")
        await page.select_option("[data-testid='audit-time-range']", "24h")
        
        await page.click("[data-testid='apply-audit-filter']")
        await page.wait_for_timeout(2000)
        
        # Verify filtered results
        audit_entries = await page.query_selector_all("[data-testid='audit-entry']")
        assert len(audit_entries) > 0
        
        # Click on audit entry for details
        if audit_entries:
            await audit_entries[0].click()
            await page.wait_for_selector("[data-testid='audit-details-modal']")
            
            audit_details = await page.text_content("[data-testid='audit-details']")
            assert "user" in audit_details.lower()
            
            await page.click("[data-testid='close-modal']")
        
        # Take screenshot of user management
        await page.screenshot(path="testing/reports/screenshots/user_management.png")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        authenticated_page: Page
    ):
        """Test error handling and recovery workflows"""
        
        page = authenticated_page
        
        # Step 1: Test network error handling
        # Simulate network disconnection
        await page.route("**/api/v1/**", lambda route: route.abort())
        
        # Try to perform action that requires API
        await page.click("[data-testid='nav-policies']")
        await page.wait_for_selector("[data-testid='error-message']", timeout=10000)
        
        error_message = await page.text_content("[data-testid='error-message']")
        assert any(keyword in error_message.lower() for keyword in ['error', 'failed', 'network'])
        
        # Verify retry button exists
        retry_button = await page.query_selector("[data-testid='retry-button']")
        assert retry_button is not None
        
        # Restore network
        await page.unroute("**/api/v1/**")
        
        # Click retry
        await retry_button.click()
        await page.wait_for_selector("[data-testid='policies-page']", timeout=15000)
        
        # Step 2: Test form validation errors
        await page.click("[data-testid='create-policy-button']")
        await page.wait_for_selector("[data-testid='policy-wizard']")
        
        # Submit form without required fields
        await page.click("[data-testid='next-step']")
        
        # Verify validation errors appear
        await page.wait_for_selector("[data-testid='validation-error']")
        
        validation_errors = await page.query_selector_all("[data-testid='validation-error']")
        assert len(validation_errors) > 0
        
        # Fill required fields and continue
        await page.fill("[data-testid='policy-name']", "Error Test Policy")
        await page.fill("[data-testid='policy-description']", "Testing error handling")
        await page.select_option("[data-testid='policy-category']", "security")
        
        await page.click("[data-testid='next-step']")
        await page.wait_for_selector("[data-testid='rule-builder']")
        
        # Step 3: Test session timeout handling
        # Clear session storage to simulate timeout
        await page.evaluate("window.sessionStorage.clear()")
        await page.evaluate("window.localStorage.clear()")
        
        # Try to perform authenticated action
        await page.click("[data-testid='next-step']")
        
        # Should redirect to login
        await page.wait_for_selector("[data-testid='login-form']", timeout=15000)
        
        # Verify session timeout message
        timeout_message = await page.query_selector("[data-testid='session-timeout-message']")
        if timeout_message:
            message_text = await timeout_message.text_content()
            assert "session" in message_text.lower()
        
        # Re-authenticate
        await self._authenticate_user(page, {"test_user_email": "test@example.com", "test_user_password": "TestPassword123!"})
        
        # Step 4: Test graceful degradation
        # Simulate partial service failure
        await page.route("**/api/v1/ai/**", lambda route: route.fulfill(status=503))
        
        # Navigate to AI assistant
        await page.click("[data-testid='nav-ai-assistant']")
        
        # Verify degraded functionality message
        await page.wait_for_selector("[data-testid='service-degraded-message']", timeout=10000)
        
        degraded_message = await page.text_content("[data-testid='service-degraded-message']")
        assert "temporarily unavailable" in degraded_message.lower() or "limited functionality" in degraded_message.lower()
        
        # Verify other features still work
        await page.click("[data-testid='nav-dashboard']")
        await page.wait_for_selector("[data-testid='dashboard']")
        
        # Restore AI service
        await page.unroute("**/api/v1/ai/**")
        
        # Take screenshot of error handling
        await page.screenshot(path="testing/reports/screenshots/error_handling.png")
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_under_load(
        self,
        authenticated_page: Page
    ):
        """Test application performance under simulated load"""
        
        page = authenticated_page
        
        # Step 1: Measure page load times
        start_time = asyncio.get_event_loop().time()
        
        await page.click("[data-testid='nav-analytics']")
        await page.wait_for_selector("[data-testid='analytics-dashboard']")
        
        load_time = asyncio.get_event_loop().time() - start_time
        assert load_time < 5.0  # Should load within 5 seconds
        
        # Step 2: Test with large data sets
        await page.click("[data-testid='load-test-data-button']")  # If available
        await page.wait_for_timeout(2000)
        
        # Scroll through large list to test rendering performance
        for _ in range(10):
            await page.keyboard.press("PageDown")
            await page.wait_for_timeout(100)
        
        # Step 3: Test rapid interactions
        for i in range(20):
            await page.click(f"[data-testid='quick-action-{i % 5}']")
            await page.wait_for_timeout(50)
        
        # Step 4: Monitor memory usage (if browser supports it)
        try:
            memory_info = await page.evaluate("performance.memory")
            if memory_info:
                assert memory_info["usedJSHeapSize"] < 100 * 1024 * 1024  # Less than 100MB
        except:
            pass  # Memory API not available in all browsers
        
        # Step 5: Test concurrent operations
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(page.click("[data-testid='refresh-data-button']"))
            tasks.append(task)
            await asyncio.sleep(0.1)  # Slight delay between clicks
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify application is still responsive
        await page.click("[data-testid='nav-dashboard']")
        await page.wait_for_selector("[data-testid='dashboard']", timeout=10000)
        
        # Take screenshot of performance test
        await page.screenshot(path="testing/reports/screenshots/performance_test.png")