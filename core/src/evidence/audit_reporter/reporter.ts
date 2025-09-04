/**
 * Audit Reporter Service for PolicyCortex PROVE Pillar
 * Generates board-ready PDF reports with QR codes and digital signatures
 */

import { createHash } from 'crypto';
import * as QRCode from 'qrcode';
import PDFDocument from 'pdfkit';
import { sign, verify, generateKeyPairSync } from 'crypto';
import { readFileSync, writeFileSync, createWriteStream } from 'fs';
import { join } from 'path';
import axios from 'axios';

// Interfaces
interface Evidence {
  id: string;
  hash: string;
  timestamp: string;
  eventType: string;
  resourceId: string;
  policyId: string;
  policyName: string;
  complianceStatus: string;
  actor: string;
  subscriptionId: string;
  resourceGroup: string;
  resourceType: string;
  details: Record<string, any>;
  metadata: Record<string, string>;
  blockIndex?: number;
  blockHash?: string;
  immutable?: boolean;
}

interface ChainVerification {
  isValid: boolean;
  totalBlocks: number;
  totalEvents: number;
  invalidBlocks: number[];
  verificationTimestamp: string;
  chainHash: string;
}

interface ComplianceSummary {
  subscriptionId: string;
  period: string;
  complianceScore: number;
  totalChecks: number;
  complianceBreakdown: {
    [status: string]: {
      count: number;
      resourceCount: number;
      policyCount: number;
    };
  };
}

interface ReportOptions {
  format: 'PDF' | 'SSP' | 'POAM';
  includeQRCode: boolean;
  digitalSignature: boolean;
  includeEvidence: boolean;
  evidenceLimit?: number;
  startDate?: string;
  endDate?: string;
}

interface ReportMetadata {
  reportId: string;
  generatedAt: string;
  generatedBy: string;
  reportType: string;
  evidenceCount: number;
  chainStatus: ChainVerification;
  signature?: string;
  publicKey?: string;
}

export class AuditReporter {
  private apiBaseUrl: string;
  private privateKey: string;
  private publicKey: string;

  constructor(apiBaseUrl: string = 'http://localhost:8080') {
    this.apiBaseUrl = apiBaseUrl;
    
    // Generate or load key pair for digital signatures
    const keyPair = this.loadOrGenerateKeyPair();
    this.privateKey = keyPair.privateKey;
    this.publicKey = keyPair.publicKey;
  }

  /**
   * Load existing key pair or generate new one
   */
  private loadOrGenerateKeyPair(): { privateKey: string; publicKey: string } {
    try {
      // Try to load existing keys
      const privateKey = readFileSync('private_key.pem', 'utf8');
      const publicKey = readFileSync('public_key.pem', 'utf8');
      return { privateKey, publicKey };
    } catch {
      // Generate new key pair
      const { privateKey, publicKey } = generateKeyPairSync('rsa', {
        modulusLength: 4096,
        publicKeyEncoding: {
          type: 'spki',
          format: 'pem'
        },
        privateKeyEncoding: {
          type: 'pkcs8',
          format: 'pem'
        }
      });

      // Save keys for future use
      writeFileSync('private_key.pem', privateKey);
      writeFileSync('public_key.pem', publicKey);

      return { privateKey, publicKey };
    }
  }

  /**
   * Generate comprehensive audit report
   */
  async generateReport(
    subscriptionId: string,
    options: ReportOptions
  ): Promise<{ reportPath: string; metadata: ReportMetadata }> {
    // Fetch data
    const [evidence, chainStatus, complianceSummary] = await Promise.all([
      this.fetchEvidence(subscriptionId, options),
      this.fetchChainStatus(),
      this.fetchComplianceSummary(subscriptionId)
    ]);

    const reportId = this.generateReportId();
    const metadata: ReportMetadata = {
      reportId,
      generatedAt: new Date().toISOString(),
      generatedBy: 'PolicyCortex PROVE System',
      reportType: options.format,
      evidenceCount: evidence.length,
      chainStatus
    };

    let reportPath: string;

    switch (options.format) {
      case 'PDF':
        reportPath = await this.generatePDFReport(
          evidence,
          chainStatus,
          complianceSummary,
          metadata,
          options
        );
        break;
      case 'SSP':
        reportPath = await this.generateSSPReport(
          evidence,
          chainStatus,
          complianceSummary,
          metadata
        );
        break;
      case 'POAM':
        reportPath = await this.generatePOAMReport(
          evidence,
          chainStatus,
          complianceSummary,
          metadata
        );
        break;
      default:
        throw new Error(`Unsupported report format: ${options.format}`);
    }

    // Add digital signature if requested
    if (options.digitalSignature) {
      const signature = this.signReport(reportPath);
      metadata.signature = signature;
      metadata.publicKey = this.publicKey;
    }

    return { reportPath, metadata };
  }

  /**
   * Generate PDF report with QR codes
   */
  private async generatePDFReport(
    evidence: Evidence[],
    chainStatus: ChainVerification,
    complianceSummary: ComplianceSummary,
    metadata: ReportMetadata,
    options: ReportOptions
  ): Promise<string> {
    const doc = new PDFDocument({
      size: 'A4',
      margins: {
        top: 50,
        bottom: 50,
        left: 50,
        right: 50
      },
      info: {
        Title: 'PolicyCortex Compliance Audit Report',
        Author: 'PolicyCortex PROVE System',
        Subject: 'Immutable Evidence Chain Report',
        Keywords: 'audit compliance evidence blockchain'
      }
    });

    const reportPath = `reports/audit_report_${metadata.reportId}.pdf`;
    const stream = createWriteStream(reportPath);
    doc.pipe(stream);

    // Title Page
    doc.fontSize(24).text('PolicyCortex Compliance Audit Report', {
      align: 'center'
    });
    doc.moveDown();
    doc.fontSize(12).text(`Report ID: ${metadata.reportId}`, { align: 'center' });
    doc.text(`Generated: ${new Date(metadata.generatedAt).toLocaleString()}`, { align: 'center' });
    doc.moveDown();

    // Add QR code for verification
    if (options.includeQRCode) {
      const verificationUrl = `${this.apiBaseUrl}/api/v1/evidence/verify/${metadata.reportId}`;
      const qrCodeDataUrl = await QRCode.toDataURL(verificationUrl, {
        width: 150,
        margin: 1
      });
      
      const qrBuffer = Buffer.from(qrCodeDataUrl.split(',')[1], 'base64');
      doc.image(qrBuffer, doc.page.width / 2 - 75, doc.y, {
        width: 150,
        height: 150
      });
      doc.moveDown(8);
      doc.fontSize(10).text('Scan to verify report integrity', { align: 'center' });
    }

    // Executive Summary
    doc.addPage();
    doc.fontSize(18).text('Executive Summary', { underline: true });
    doc.moveDown();
    doc.fontSize(12);
    doc.text(`Subscription: ${complianceSummary.subscriptionId}`);
    doc.text(`Compliance Score: ${complianceSummary.complianceScore.toFixed(2)}%`);
    doc.text(`Total Policy Checks: ${complianceSummary.totalChecks}`);
    doc.text(`Reporting Period: ${complianceSummary.period}`);
    doc.moveDown();

    // Chain Verification Status
    doc.fontSize(16).text('Evidence Chain Status', { underline: true });
    doc.moveDown();
    doc.fontSize(12);
    doc.text(`Chain Valid: ${chainStatus.isValid ? 'YES ✓' : 'NO ✗'}`);
    doc.text(`Total Blocks: ${chainStatus.totalBlocks}`);
    doc.text(`Total Evidence Records: ${chainStatus.totalEvents}`);
    doc.text(`Chain Hash: ${chainStatus.chainHash}`);
    doc.text(`Verification Timestamp: ${new Date(chainStatus.verificationTimestamp).toLocaleString()}`);
    
    if (chainStatus.invalidBlocks.length > 0) {
      doc.fillColor('red');
      doc.text(`Invalid Blocks: ${chainStatus.invalidBlocks.join(', ')}`);
      doc.fillColor('black');
    }
    doc.moveDown();

    // Compliance Breakdown
    doc.fontSize(16).text('Compliance Breakdown', { underline: true });
    doc.moveDown();
    doc.fontSize(12);
    
    for (const [status, data] of Object.entries(complianceSummary.complianceBreakdown)) {
      const color = this.getStatusColor(status);
      doc.fillColor(color);
      doc.text(`${status}: ${data.count} checks across ${data.resourceCount} resources and ${data.policyCount} policies`);
    }
    doc.fillColor('black');
    doc.moveDown();

    // Evidence Details (if requested)
    if (options.includeEvidence && evidence.length > 0) {
      doc.addPage();
      doc.fontSize(18).text('Evidence Details', { underline: true });
      doc.moveDown();
      
      const evidenceToInclude = evidence.slice(0, options.evidenceLimit || 100);
      
      for (const ev of evidenceToInclude) {
        doc.fontSize(10);
        doc.text(`ID: ${ev.id}`);
        doc.text(`Time: ${new Date(ev.timestamp).toLocaleString()}`);
        doc.text(`Resource: ${ev.resourceId}`);
        doc.text(`Policy: ${ev.policyName} (${ev.policyId})`);
        doc.text(`Status: ${ev.complianceStatus}`);
        doc.text(`Hash: ${ev.hash}`);
        
        if (ev.blockIndex !== undefined) {
          doc.text(`Block: ${ev.blockIndex} (${ev.blockHash})`);
          doc.text(`Immutable: ${ev.immutable ? 'YES' : 'NO'}`);
        }
        
        doc.moveDown(0.5);
        doc.strokeColor('#cccccc').moveTo(50, doc.y).lineTo(doc.page.width - 50, doc.y).stroke();
        doc.moveDown(0.5);
        
        // Add page break if needed
        if (doc.y > doc.page.height - 100) {
          doc.addPage();
        }
      }
    }

    // Digital Signature Page
    if (options.digitalSignature) {
      doc.addPage();
      doc.fontSize(18).text('Digital Signature', { underline: true });
      doc.moveDown();
      doc.fontSize(10);
      doc.text('This document has been digitally signed by the PolicyCortex PROVE system.');
      doc.text('Public Key for Verification:');
      doc.fontSize(8).font('Courier');
      doc.text(this.publicKey);
      doc.font('Helvetica');
    }

    // Finalize PDF
    doc.end();
    
    return new Promise((resolve) => {
      stream.on('finish', () => resolve(reportPath));
    });
  }

  /**
   * Generate System Security Plan (SSP) report
   */
  private async generateSSPReport(
    evidence: Evidence[],
    chainStatus: ChainVerification,
    complianceSummary: ComplianceSummary,
    metadata: ReportMetadata
  ): Promise<string> {
    const ssp = {
      systemName: 'PolicyCortex Governance Platform',
      systemIdentifier: metadata.reportId,
      responsibleOrganization: 'Organization Name',
      systemOwner: 'System Owner',
      authorizedUsers: ['Admin', 'Auditor', 'Operator'],
      systemDescription: 'AI-Powered Azure Governance Platform with immutable evidence chain',
      systemBoundary: {
        components: [
          'Azure Resources',
          'Policy Engine',
          'Evidence Collector',
          'Hash Chain',
          'Audit Reporter'
        ]
      },
      securityControls: this.mapEvidenceToControls(evidence),
      complianceStatus: complianceSummary,
      evidenceChain: {
        status: chainStatus,
        totalEvidence: evidence.length,
        immutableRecords: evidence.filter(e => e.immutable).length
      },
      generatedAt: metadata.generatedAt,
      signature: metadata.signature
    };

    const reportPath = `reports/ssp_${metadata.reportId}.json`;
    writeFileSync(reportPath, JSON.stringify(ssp, null, 2));
    
    return reportPath;
  }

  /**
   * Generate Plan of Action and Milestones (POA&M) report
   */
  private async generatePOAMReport(
    evidence: Evidence[],
    chainStatus: ChainVerification,
    complianceSummary: ComplianceSummary,
    metadata: ReportMetadata
  ): Promise<string> {
    // Filter non-compliant evidence for POA&M
    const nonCompliantEvidence = evidence.filter(
      e => e.complianceStatus === 'NonCompliant' || e.complianceStatus === 'Warning'
    );

    const poam = {
      systemName: 'PolicyCortex Governance Platform',
      poamId: metadata.reportId,
      generatedAt: metadata.generatedAt,
      findings: nonCompliantEvidence.map((ev, index) => ({
        itemNumber: index + 1,
        weakness: `Policy violation: ${ev.policyName}`,
        pocOrganization: 'IT Security',
        resources: ev.resourceId,
        scheduledCompletionDate: this.calculateRemediationDate(ev),
        milestones: this.generateMilestones(ev),
        status: 'Ongoing',
        comments: `Evidence Hash: ${ev.hash}`,
        evidenceId: ev.id,
        blockIndex: ev.blockIndex
      })),
      summary: {
        totalFindings: nonCompliantEvidence.length,
        complianceScore: complianceSummary.complianceScore,
        chainValid: chainStatus.isValid
      },
      signature: metadata.signature
    };

    const reportPath = `reports/poam_${metadata.reportId}.json`;
    writeFileSync(reportPath, JSON.stringify(poam, null, 2));
    
    return reportPath;
  }

  /**
   * Fetch evidence from the API
   */
  private async fetchEvidence(
    subscriptionId: string,
    options: ReportOptions
  ): Promise<Evidence[]> {
    try {
      const response = await axios.get(
        `${this.apiBaseUrl}/api/v1/evidence/subscription/${subscriptionId}`,
        {
          params: {
            limit: options.evidenceLimit || 1000,
            startDate: options.startDate,
            endDate: options.endDate
          }
        }
      );
      return response.data.evidence || [];
    } catch (error) {
      console.error('Error fetching evidence:', error);
      return [];
    }
  }

  /**
   * Fetch chain verification status
   */
  private async fetchChainStatus(): Promise<ChainVerification> {
    try {
      const response = await axios.get(`${this.apiBaseUrl}/api/v1/evidence/chain`);
      return response.data;
    } catch (error) {
      console.error('Error fetching chain status:', error);
      return {
        isValid: false,
        totalBlocks: 0,
        totalEvents: 0,
        invalidBlocks: [],
        verificationTimestamp: new Date().toISOString(),
        chainHash: ''
      };
    }
  }

  /**
   * Fetch compliance summary
   */
  private async fetchComplianceSummary(subscriptionId: string): Promise<ComplianceSummary> {
    try {
      const response = await axios.get(
        `${this.apiBaseUrl}/api/v1/evidence/compliance/summary/${subscriptionId}`
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching compliance summary:', error);
      return {
        subscriptionId,
        period: '30_days',
        complianceScore: 0,
        totalChecks: 0,
        complianceBreakdown: {}
      };
    }
  }

  /**
   * Sign report file
   */
  private signReport(reportPath: string): string {
    const reportContent = readFileSync(reportPath);
    const signer = sign('sha256', reportContent, this.privateKey);
    return signer.toString('base64');
  }

  /**
   * Verify report signature
   */
  verifyReportSignature(reportPath: string, signature: string, publicKey: string): boolean {
    try {
      const reportContent = readFileSync(reportPath);
      const verifier = verify(
        'sha256',
        reportContent,
        publicKey,
        Buffer.from(signature, 'base64')
      );
      return verifier;
    } catch (error) {
      console.error('Error verifying signature:', error);
      return false;
    }
  }

  /**
   * Generate unique report ID
   */
  private generateReportId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(7);
    return `${timestamp}-${random}`;
  }

  /**
   * Get color for compliance status
   */
  private getStatusColor(status: string): string {
    switch (status.toLowerCase()) {
      case 'compliant':
        return 'green';
      case 'noncompliant':
      case 'non-compliant':
        return 'red';
      case 'warning':
        return 'orange';
      case 'error':
        return 'red';
      default:
        return 'black';
    }
  }

  /**
   * Map evidence to security controls
   */
  private mapEvidenceToControls(evidence: Evidence[]): any[] {
    const controlsMap = new Map();
    
    evidence.forEach(ev => {
      const controlId = this.deriveControlId(ev.policyId);
      if (!controlsMap.has(controlId)) {
        controlsMap.set(controlId, {
          controlId,
          title: ev.policyName,
          status: ev.complianceStatus,
          evidence: []
        });
      }
      controlsMap.get(controlId).evidence.push(ev.id);
    });
    
    return Array.from(controlsMap.values());
  }

  /**
   * Derive control ID from policy ID
   */
  private deriveControlId(policyId: string): string {
    // Map policy IDs to control families (simplified)
    if (policyId.includes('access')) return 'AC-1';
    if (policyId.includes('audit')) return 'AU-1';
    if (policyId.includes('config')) return 'CM-1';
    if (policyId.includes('incident')) return 'IR-1';
    if (policyId.includes('security')) return 'SC-1';
    return 'XX-1';
  }

  /**
   * Calculate remediation date
   */
  private calculateRemediationDate(evidence: Evidence): string {
    const date = new Date();
    // Add 30 days for remediation
    date.setDate(date.getDate() + 30);
    return date.toISOString();
  }

  /**
   * Generate milestones for remediation
   */
  private generateMilestones(evidence: Evidence): any[] {
    return [
      {
        milestone: 'Initial Assessment',
        scheduledDate: new Date().toISOString(),
        status: 'Complete'
      },
      {
        milestone: 'Remediation Plan Development',
        scheduledDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'In Progress'
      },
      {
        milestone: 'Implementation',
        scheduledDate: new Date(Date.now() + 21 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'Pending'
      },
      {
        milestone: 'Validation',
        scheduledDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'Pending'
      }
    ];
  }
}

// Export for use in other modules
export default AuditReporter;