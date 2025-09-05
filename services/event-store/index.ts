/**
 * PolicyCortex Event Store Service (T04)
 * In-memory event store with append and replay capabilities
 * For production, this would use PostgreSQL as shown in the Rust implementation
 */

import type { Event } from '../../packages/types/src/events';

export interface StoredEvent {
  id: number;
  timestamp: string;
  payload: Event;
}

export interface EventStoreOptions {
  maxEvents?: number;
  persistPath?: string;
}

export class EventStore {
  private events: StoredEvent[] = [];
  private nextId = 1;
  private options: EventStoreOptions;

  constructor(options: EventStoreOptions = {}) {
    this.options = {
      maxEvents: options.maxEvents || 10000,
      persistPath: options.persistPath
    };
    
    // Load persisted events if path provided
    if (this.options.persistPath) {
      this.loadFromDisk();
    }
  }

  /**
   * Append an event to the store
   */
  async append(event: Event): Promise<StoredEvent> {
    const stored: StoredEvent = {
      id: this.nextId++,
      timestamp: new Date().toISOString(),
      payload: event
    };

    this.events.push(stored);

    // Trim if exceeds max
    if (this.events.length > this.options.maxEvents!) {
      this.events = this.events.slice(-this.options.maxEvents!);
    }

    // Persist if configured
    if (this.options.persistPath) {
      await this.persistToDisk();
    }

    return stored;
  }

  /**
   * List all events in order
   */
  async list(): Promise<StoredEvent[]> {
    return [...this.events];
  }

  /**
   * Replay events for reconstruction
   * Returns raw event payloads in order
   */
  async replay(): Promise<Event[]> {
    return this.events.map(e => e.payload);
  }

  /**
   * Get events by type
   */
  async getByType<T extends Event>(type: T['type']): Promise<T[]> {
    return this.events
      .filter(e => e.payload.type === type)
      .map(e => e.payload as T);
  }

  /**
   * Get events in time range
   */
  async getByTimeRange(start: Date, end: Date): Promise<StoredEvent[]> {
    const startTime = start.getTime();
    const endTime = end.getTime();
    
    return this.events.filter(e => {
      const eventTime = new Date(e.timestamp).getTime();
      return eventTime >= startTime && eventTime <= endTime;
    });
  }

  /**
   * Clear all events (for testing)
   */
  async clear(): Promise<void> {
    this.events = [];
    this.nextId = 1;
    
    if (this.options.persistPath) {
      await this.persistToDisk();
    }
  }

  /**
   * Get count of events
   */
  count(): number {
    return this.events.length;
  }

  /**
   * Persist to disk (simple JSON for demo)
   */
  private async persistToDisk(): Promise<void> {
    if (!this.options.persistPath) return;
    
    try {
      const fs = await import('fs/promises');
      const data = JSON.stringify({
        events: this.events,
        nextId: this.nextId
      }, null, 2);
      
      await fs.writeFile(this.options.persistPath, data, 'utf-8');
    } catch (error) {
      console.error('Failed to persist events:', error);
    }
  }

  /**
   * Load from disk
   */
  private loadFromDisk(): void {
    if (!this.options.persistPath) return;
    
    try {
      const fs = require('fs');
      if (fs.existsSync(this.options.persistPath)) {
        const data = fs.readFileSync(this.options.persistPath, 'utf-8');
        const parsed = JSON.parse(data);
        this.events = parsed.events || [];
        this.nextId = parsed.nextId || 1;
      }
    } catch (error) {
      console.error('Failed to load events:', error);
    }
  }
}

// Singleton instance for the application
let globalStore: EventStore | null = null;

export function getEventStore(options?: EventStoreOptions): EventStore {
  if (!globalStore) {
    globalStore = new EventStore(options);
  }
  return globalStore;
}

// Export for testing
export function resetEventStore(): void {
  globalStore = null;
}