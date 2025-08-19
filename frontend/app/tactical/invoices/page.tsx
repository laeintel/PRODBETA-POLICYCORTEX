'use client';

import { useState, useMemo } from 'react';
import TacticalPageTemplate from '../../../components/TacticalPageTemplate';
import { FileText, DollarSign, Download, Eye, CheckCircle, AlertTriangle, Clock, Search, Filter, Plus, Upload, CreditCard } from 'lucide-react';

export default function Page() {
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [selectedPeriod, setSelectedPeriod] = useState('current');
  const [searchTerm, setSearchTerm] = useState('');

  // Mock invoice data
  const invoices = [
    {
      id: 'INV-2024-001',
      subscription: 'Azure Production',
      subscriptionId: '205b477d-17e7-4b3b-92c1-32cf02626b78',
      billingPeriod: 'August 2024',
      amount: 12450.80,
      dueDate: '2024-09-15',
      issueDate: '2024-08-15',
      status: 'paid',
      paymentDate: '2024-08-18',
      services: [
        { name: 'Virtual Machines', cost: 4200.50 },
        { name: 'Azure SQL Database', cost: 3100.20 },
        { name: 'Storage Account', cost: 1850.30 },
        { name: 'Application Gateway', cost: 980.45 },
        { name: 'Other Services', cost: 2319.35 }
      ],
      tags: ['production', 'critical'],
      department: 'Engineering'
    },
    {
      id: 'INV-2024-002',
      subscription: 'Azure Development',
      subscriptionId: '9ef5b184-d371-462a-bc75-5024ce8baff7',
      billingPeriod: 'August 2024',
      amount: 4200.15,
      dueDate: '2024-09-15',
      issueDate: '2024-08-15',
      status: 'pending',
      services: [
        { name: 'Virtual Machines', cost: 1500.25 },
        { name: 'Azure Functions', cost: 850.50 },
        { name: 'Storage Account', cost: 920.40 },
        { name: 'Other Services', cost: 929.00 }
      ],
      tags: ['development', 'testing'],
      department: 'Engineering'
    },
    {
      id: 'INV-2024-003',
      subscription: 'Azure Analytics',
      subscriptionId: '1ecc95d1-e5bb-43e2-9324-30a17cb6b01c',
      billingPeriod: 'August 2024',
      amount: 8750.90,
      dueDate: '2024-09-15',
      issueDate: '2024-08-15',
      status: 'overdue',
      services: [
        { name: 'Azure Synapse', cost: 3200.60 },
        { name: 'Data Factory', cost: 2100.45 },
        { name: 'Cosmos DB', cost: 1850.25 },
        { name: 'Power BI Premium', cost: 1599.60 }
      ],
      tags: ['analytics', 'data'],
      department: 'Data Science',
      overdueBy: 5
    },
    {
      id: 'INV-2024-004',
      subscription: 'Azure Security',
      subscriptionId: 'sec-001-abc-def-ghi',
      billingPeriod: 'August 2024',
      amount: 2800.45,
      dueDate: '2024-09-15',
      issueDate: '2024-08-15',
      status: 'disputed',
      services: [
        { name: 'Azure Sentinel', cost: 1200.25 },
        { name: 'Key Vault', cost: 450.80 },
        { name: 'Azure AD Premium', cost: 850.40 },
        { name: 'Other Services', cost: 299.00 }
      ],
      tags: ['security', 'compliance'],
      department: 'Security',
      disputeReason: 'Unexpected charges for unused services'
    }
  ];

  const invoiceStats = useMemo(() => {
    const totalAmount = invoices.reduce((sum, inv) => sum + inv.amount, 0);
    const paidAmount = invoices.filter(inv => inv.status === 'paid').reduce((sum, inv) => sum + inv.amount, 0);
    const pendingAmount = invoices.filter(inv => inv.status === 'pending').reduce((sum, inv) => sum + inv.amount, 0);
    const overdueAmount = invoices.filter(inv => inv.status === 'overdue').reduce((sum, inv) => sum + inv.amount, 0);
    const disputedAmount = invoices.filter(inv => inv.status === 'disputed').reduce((sum, inv) => sum + inv.amount, 0);

    const paid = invoices.filter(inv => inv.status === 'paid').length;
    const pending = invoices.filter(inv => inv.status === 'pending').length;
    const overdue = invoices.filter(inv => inv.status === 'overdue').length;
    const disputed = invoices.filter(inv => inv.status === 'disputed').length;

    return {
      totalAmount, paidAmount, pendingAmount, overdueAmount, disputedAmount,
      total: invoices.length, paid, pending, overdue, disputed
    };
  }, [invoices]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'paid': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'overdue': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'disputed': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'paid': return 'bg-green-900/50 text-green-300 border-green-500/30';
      case 'pending': return 'bg-yellow-900/50 text-yellow-300 border-yellow-500/30';
      case 'overdue': return 'bg-red-900/50 text-red-300 border-red-500/30';
      case 'disputed': return 'bg-orange-900/50 text-orange-300 border-orange-500/30';
      default: return 'bg-gray-900/50 text-gray-300 border-gray-500/30';
    }
  };

  const content = (
    <div className="space-y-8">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search invoices..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-2 text-white focus:ring-2 focus:ring-blue-500 w-64"
            />
          </div>
          <select 
            value={selectedPeriod} 
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="current">Current Month</option>
            <option value="last">Last Month</option>
            <option value="quarter">This Quarter</option>
            <option value="year">This Year</option>
          </select>
          <select 
            value={selectedStatus} 
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="paid">Paid</option>
            <option value="pending">Pending</option>
            <option value="overdue">Overdue</option>
            <option value="disputed">Disputed</option>
          </select>
        </div>
        <div className="flex items-center space-x-3">
          <button className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Upload className="w-4 h-4" />
            <span>Import Invoices</span>
          </button>
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white flex items-center space-x-2 transition-colors">
            <Plus className="w-4 h-4" />
            <span>Manual Invoice</span>
          </button>
        </div>
      </div>

      {/* Invoice Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 backdrop-blur-md rounded-xl border border-blue-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <FileText className="w-8 h-8 text-blue-400" />
            <div className="text-sm text-blue-300">{invoiceStats.total}</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${invoiceStats.totalAmount.toLocaleString()}</p>
          <p className="text-blue-300 text-sm">Total Amount</p>
        </div>

        <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 backdrop-blur-md rounded-xl border border-green-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <div className="text-sm text-green-300">{invoiceStats.paid}</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${invoiceStats.paidAmount.toLocaleString()}</p>
          <p className="text-green-300 text-sm">Paid</p>
        </div>

        <div className="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 backdrop-blur-md rounded-xl border border-yellow-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <Clock className="w-8 h-8 text-yellow-400" />
            <div className="text-sm text-yellow-300">{invoiceStats.pending}</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${invoiceStats.pendingAmount.toLocaleString()}</p>
          <p className="text-yellow-300 text-sm">Pending</p>
        </div>

        <div className="bg-gradient-to-br from-red-900/50 to-red-800/30 backdrop-blur-md rounded-xl border border-red-500/30 p-6">
          <div className="flex items-center justify-between mb-4">
            <AlertTriangle className="w-8 h-8 text-red-400" />
            <div className="text-sm text-red-300">{invoiceStats.overdue + invoiceStats.disputed}</div>
          </div>
          <p className="text-3xl font-bold text-white mb-2">${(invoiceStats.overdueAmount + invoiceStats.disputedAmount).toLocaleString()}</p>
          <p className="text-red-300 text-sm">Issues</p>
        </div>
      </div>

      {/* Invoice List */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <h3 className="text-xl font-bold text-white">Invoice Management</h3>
          <div className="flex items-center space-x-2">
            <button className="bg-gray-800 hover:bg-gray-700 px-3 py-1 rounded-md text-sm text-white transition-colors">
              <Filter className="w-4 h-4" />
            </button>
            <button className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded-md text-sm text-white transition-colors">
              Export All
            </button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Invoice</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Subscription</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Amount</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Due Date</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Department</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {invoices.map((invoice) => (
                <tr key={invoice.id} className="hover:bg-gray-800/30 transition-colors">
                  <td className="px-6 py-4">
                    <div>
                      <p className="text-white font-medium">{invoice.id}</p>
                      <p className="text-gray-400 text-sm">{invoice.billingPeriod}</p>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div>
                      <p className="text-white text-sm">{invoice.subscription}</p>
                      <p className="text-gray-400 text-xs">{invoice.subscriptionId.substring(0, 20)}...</p>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-white font-bold text-lg">${invoice.amount.toLocaleString()}</p>
                    <p className="text-gray-400 text-xs">{invoice.services.length} services</p>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-white text-sm">{new Date(invoice.dueDate).toLocaleDateString()}</p>
                    {invoice.status === 'overdue' && invoice.overdueBy && (
                      <p className="text-red-400 text-xs">{invoice.overdueBy} days overdue</p>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(invoice.status)}
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(invoice.status)}`}>
                        {invoice.status}
                      </span>
                    </div>
                    {invoice.status === 'paid' && invoice.paymentDate && (
                      <p className="text-green-400 text-xs mt-1">Paid {new Date(invoice.paymentDate).toLocaleDateString()}</p>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <div>
                      <p className="text-white text-sm">{invoice.department}</p>
                      <div className="flex items-center space-x-1 mt-1">
                        {invoice.tags.map((tag, index) => (
                          <span key={index} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-blue-900/50 text-blue-300 border border-blue-500/30">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-2">
                      <button className="text-blue-400 hover:text-blue-300 flex items-center space-x-1">
                        <Eye className="w-4 h-4" />
                        <span className="text-sm">View</span>
                      </button>
                      <button className="text-gray-400 hover:text-gray-300 flex items-center space-x-1">
                        <Download className="w-4 h-4" />
                        <span className="text-sm">PDF</span>
                      </button>
                      {invoice.status === 'pending' && (
                        <button className="text-green-400 hover:text-green-300 flex items-center space-x-1">
                          <CreditCard className="w-4 h-4" />
                          <span className="text-sm">Pay</span>
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Service Breakdown and Payment Timeline */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4">Service Breakdown - Production</h3>
          <div className="space-y-4">
            {invoices[0].services.map((service, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                <div>
                  <p className="text-white font-medium">{service.name}</p>
                  <p className="text-gray-400 text-sm">Monthly usage</p>
                </div>
                <div className="text-right">
                  <p className="text-white font-bold">${service.cost.toLocaleString()}</p>
                  <p className="text-gray-400 text-xs">{((service.cost / invoices[0].amount) * 100).toFixed(1)}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-bold text-white mb-4">Payment Timeline</h3>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-green-500 rounded-full" />
              <div className="flex-1">
                <p className="text-white text-sm">Invoice Generated</p>
                <p className="text-gray-400 text-xs">August 15, 2024</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-blue-500 rounded-full" />
              <div className="flex-1">
                <p className="text-white text-sm">Payment Processed</p>
                <p className="text-gray-400 text-xs">August 18, 2024</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-3 h-3 bg-yellow-500 rounded-full" />
              <div className="flex-1">
                <p className="text-white text-sm">Due Date</p>
                <p className="text-gray-400 text-xs">September 15, 2024</p>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="text-green-300 font-medium">Payment Complete</span>
            </div>
            <p className="text-green-200 text-sm">Automatic payment processed successfully via credit card ending in 4532</p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <TacticalPageTemplate 
      title="Invoice Management" 
      subtitle="Automated Invoice Processing & Payment Tracking" 
      icon={FileText}
    >
      {content}
    </TacticalPageTemplate>
  );
}