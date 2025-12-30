import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ModalService } from '../../shared/modal/modal.service';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-reports',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="min-h-screen bg-gray-50">
      <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div class="px-4 py-6 sm:px-0">
          <h1 class="text-3xl font-bold text-gray-900 mb-8">Reporting System</h1>
          
          <!-- System Summary -->
          <div class="bg-white shadow rounded-lg p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">System Summary</h2>
            <div *ngIf="loading" class="flex items-center justify-center py-8">
              <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span class="ml-3 text-gray-600">Loading reporting system...</span>
            </div>
            
            <div *ngIf="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
              <div class="flex">
                <div class="flex-shrink-0">
                  <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                  </svg>
                </div>
                <div class="ml-3">
                  <h3 class="text-sm font-medium text-red-800">Error loading data</h3>
                  <div class="mt-2 text-sm text-red-700">{{ error }}</div>
                </div>
              </div>
            </div>
            
            <div *ngIf="!loading && !error && systemSummary" class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div class="bg-blue-50 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-blue-800">Total Trades</h3>
                <p class="text-2xl font-bold text-blue-900">{{ systemSummary.report_agent?.total_trades || 0 }}</p>
              </div>
              <div class="bg-green-50 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-green-800">Forecast Errors</h3>
                <p class="text-2xl font-bold text-green-900">{{ systemSummary.report_agent?.total_forecast_errors || 0 }}</p>
              </div>
              <div class="bg-purple-50 p-4 rounded-lg">
                <h3 class="text-sm font-medium text-purple-800">Agents Tracked</h3>
                <p class="text-2xl font-bold text-purple-900">{{ systemSummary.report_agent?.agents_tracked || 0 }}</p>
              </div>
            </div>
          </div>
          
          <!-- Report Generation -->
          <div class="bg-white shadow rounded-lg p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Generate Reports</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <button 
                (click)="generateReport('daily')"
                [disabled]="generating"
                class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                {{ generating && currentReport === 'daily' ? 'Generating...' : 'Daily Report' }}
              </button>
              <button 
                (click)="generateReport('weekly')"
                [disabled]="generating"
                class="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                {{ generating && currentReport === 'weekly' ? 'Generating...' : 'Weekly Report' }}
              </button>
              <button 
                (click)="generateReport('agent-performance')"
                [disabled]="generating"
                class="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                {{ generating && currentReport === 'agent-performance' ? 'Generating...' : 'Agent Performance' }}
              </button>
              <button 
                (click)="generateReport('trade-based')"
                [disabled]="generating"
                class="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                {{ generating && currentReport === 'trade-based' ? 'Generating...' : 'Trade-Based Report' }}
              </button>
            </div>
          </div>
          
          <!-- Generated Reports -->
          <div class="bg-white shadow rounded-lg p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Generated Reports</h2>
            <div *ngIf="generatedReports.length === 0" class="text-center py-8 text-gray-500">
              No reports generated yet. Click a button above to generate your first report.
            </div>
            <div *ngIf="generatedReports.length > 0" class="space-y-4">
              <div *ngFor="let report of generatedReports" class="border border-gray-200 rounded-lg p-4">
                <div class="flex items-center justify-between">
                  <div>
                    <h3 class="text-lg font-medium text-gray-900">{{ report.report_type | titlecase }} Report</h3>
                    <p class="text-sm text-gray-500">Generated: {{ report.generated_at | date:'medium' }}</p>
                    <p class="text-sm text-gray-500">Format: {{ report.format | uppercase }}</p>
                    <p class="text-sm text-gray-500">File Size: {{ report.file_size | number }} bytes</p>
                  </div>
                  <div class="flex space-x-2">
                    <button 
                      (click)="downloadReport(report)"
                      class="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm">
                      Download
                    </button>
                    <button 
                      (click)="viewReport(report)"
                      class="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm">
                      View
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- AI Explanations -->
          <div class="bg-white shadow rounded-lg p-6">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">AI Explanations</h2>
            <div class="space-y-4">
              <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="text-lg font-medium text-gray-900 mb-2">Trade Decision Explanation</h3>
                <p class="text-sm text-gray-600 mb-4">Get AI-powered explanations for trading decisions</p>
                <button 
                  (click)="generateExplanation('trade_decision')"
                  [disabled]="explaining"
                  class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                  {{ explaining && currentExplanation === 'trade_decision' ? 'Generating...' : 'Generate Trade Explanation' }}
                </button>
              </div>
              
              <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="text-lg font-medium text-gray-900 mb-2">Agent Performance Explanation</h3>
                <p class="text-sm text-gray-600 mb-4">Get AI-powered explanations for agent performance</p>
                <button 
                  (click)="generateExplanation('agent_performance')"
                  [disabled]="explaining"
                  class="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                  {{ explaining && currentExplanation === 'agent_performance' ? 'Generating...' : 'Generate Performance Explanation' }}
                </button>
              </div>
              
              <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="text-lg font-medium text-gray-900 mb-2">Market Analysis Explanation</h3>
                <p class="text-sm text-gray-600 mb-4">Get AI-powered explanations for market analysis</p>
                <button 
                  (click)="generateExplanation('market_analysis')"
                  [disabled]="explaining"
                  class="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                  {{ explaining && currentExplanation === 'market_analysis' ? 'Generating...' : 'Generate Market Explanation' }}
                </button>
              </div>
            </div>
            
            <!-- Explanation Results -->
            <div *ngIf="explanationResult" class="mt-6 border border-gray-200 rounded-lg p-4 bg-gray-50">
              <h3 class="text-lg font-medium text-gray-900 mb-2">{{ explanationResult.title }}</h3>
              <div class="prose max-w-none">
                <p class="text-sm text-gray-700 whitespace-pre-line">{{ explanationResult.detailed_explanation }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class ReportsComponent implements OnInit {
  loading = true;
  error: string | null = null;
  systemSummary: any = null;
  generating = false;
  explaining = false;
  currentReport: string | null = null;
  currentExplanation: string | null = null;
  generatedReports: any[] = [];
  explanationResult: any = null;

  constructor(private http: HttpClient, private modalService: ModalService) {}

  ngOnInit(): void {
    this.loadSystemSummary();
  }

  loadSystemSummary(): void {
    this.loading = true;
    this.error = null;

    this.http.get<any>(`${environment.apiUrl}/reports/summary`).subscribe({
      next: (summary) => {
        this.systemSummary = summary;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading system summary:', err);
        this.error = 'Failed to load reporting system summary.';
        this.loading = false;
      }
    });
  }

  generateReport(reportType: string): void {
    this.generating = true;
    this.currentReport = reportType;

    let endpoint = '';
    switch (reportType) {
      case 'daily':
        endpoint = `${environment.apiUrl}/reports/daily`;
        break;
      case 'weekly':
        endpoint = `${environment.apiUrl}/reports/weekly`;
        break;
      case 'agent-performance':
        endpoint = `${environment.apiUrl}/reports/agent-performance`;
        break;
      default:
        this.generating = false;
        this.currentReport = null;
        return;
    }

    this.http.get<any>(endpoint).subscribe({
      next: (report) => {
        this.generatedReports.unshift(report);
        this.generating = false;
        this.currentReport = null;
      },
      error: (err) => {
        console.error(`Error generating ${reportType} report:`, err);
        this.generating = false;
        this.currentReport = null;
        this.modalService.error(`Failed to generate ${reportType} report. Please try again.`);
      }
    });
  }

  generateExplanation(explanationType: string): void {
    this.explaining = true;
    this.currentExplanation = explanationType;

    const requestData = {
      explanation_type: explanationType,
      data: this.getExplanationData(explanationType),
      tone: 'professional'
    };

    this.http.post<any>(`${environment.apiUrl}/reports/explain`, requestData).subscribe({
      next: (explanation) => {
        this.explanationResult = explanation;
        this.explaining = false;
        this.currentExplanation = null;
      },
      error: (err) => {
        console.error(`Error generating ${explanationType} explanation:`, err);
        this.explaining = false;
        this.currentExplanation = null;
        this.modalService.error(`Failed to generate ${explanationType} explanation. Please try again.`);
      }
    });
  }

  getExplanationData(explanationType: string): any {
    switch (explanationType) {
      case 'trade_decision':
        return {
          symbol: 'BTC-USD',
          trade_type: 'long',
          confidence_score: 0.75,
          entry_price: 69000.0,
          quantity: 0.001719,
          agent_signals: ['MomentumAgent', 'RiskAgent'],
          market_regime: 'bull'
        };
      case 'agent_performance':
        return {
          agent_name: 'MomentumAgent',
          win_rate: 65.2,
          total_signals: 45,
          successful_signals: 29,
          failed_signals: 16,
          avg_confidence: 0.78,
          best_performing_regime: 'bull',
          worst_performing_regime: 'sideways'
        };
      case 'market_analysis':
        return {
          market_regime: 'bull',
          volatility: 0.15,
          trend_direction: 'upward',
          regime_duration: 5,
          regime_strength: 0.8
        };
      default:
        return {};
    }
  }

  downloadReport(report: any): void {
    // In a real implementation, this would download the actual file
    this.modalService.info(`Downloading ${report.report_type} report (${report.format})`, 'Download Report');
  }

  viewReport(report: any): void {
    // In a real implementation, this would open the report in a new window
    this.modalService.info(`Viewing ${report.report_type} report (${report.format})`, 'View Report');
  }
}
