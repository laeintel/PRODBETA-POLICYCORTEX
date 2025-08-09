// Service Worker for offline support and background sync
const CACHE_NAME = 'policycortex-v1';
const DYNAMIC_CACHE = 'policycortex-dynamic-v1';
const OFFLINE_QUEUE = 'policycortex-offline-queue';

// Static assets to cache
const STATIC_ASSETS = [
  '/',
  '/offline.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/dashboard',
  '/policies',
  '/resources',
];

// API endpoints to cache
const CACHEABLE_API_PATTERNS = [
  /\/api\/v1\/metrics$/,
  /\/api\/v1\/policies$/,
  /\/api\/v1\/resources$/,
  /\/api\/v1\/compliance$/,
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS.filter(url => !url.includes('.')));
    })
  );
  self.skipWaiting();
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME && name !== DYNAMIC_CACHE)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// Fetch event - network first, fallback to cache
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests and Chrome extensions
  if (request.method !== 'GET' || url.protocol === 'chrome-extension:') {
    return;
  }

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }

  // Handle static assets and pages
  event.respondWith(handleStaticRequest(request));
});

// Handle API requests with network-first strategy
async function handleApiRequest(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Network failed, try cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log('[SW] Serving API from cache:', request.url);
      return cachedResponse;
    }
    
    // Return offline response
    return new Response(
      JSON.stringify({
        error: 'Offline',
        message: 'Request queued for sync',
        cached: false
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Handle static requests with cache-first strategy
async function handleStaticRequest(request) {
  // Check cache first
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    // Try network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_ASSETS.includes(request.url) ? CACHE_NAME : DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const offlineResponse = await caches.match('/offline.html');
      if (offlineResponse) {
        return offlineResponse;
      }
    }
    
    // Return 503 for other requests
    return new Response('Offline', { status: 503 });
  }
}

// Background sync for offline queue
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-offline-queue') {
    event.waitUntil(syncOfflineQueue());
  }
});

// Process offline queue
async function syncOfflineQueue() {
  const queue = await getOfflineQueue();
  
  for (const item of queue) {
    try {
      const response = await fetch(item.request);
      if (response.ok) {
        await removeFromQueue(item.id);
        await notifyClient(item.id, 'success', response);
      } else {
        await notifyClient(item.id, 'error', response);
      }
    } catch (error) {
      console.error('[SW] Sync failed for:', item.id, error);
    }
  }
}

// IndexedDB helpers for offline queue
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(OFFLINE_QUEUE, 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('queue')) {
        db.createObjectStore('queue', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

async function addToQueue(request) {
  const db = await openDB();
  const tx = db.transaction(['queue'], 'readwrite');
  const store = tx.objectStore('queue');
  
  const item = {
    url: request.url,
    method: request.method,
    headers: Object.fromEntries(request.headers),
    body: await request.text(),
    timestamp: Date.now()
  };
  
  return store.add(item);
}

async function getOfflineQueue() {
  const db = await openDB();
  const tx = db.transaction(['queue'], 'readonly');
  const store = tx.objectStore('queue');
  
  return new Promise((resolve, reject) => {
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function removeFromQueue(id) {
  const db = await openDB();
  const tx = db.transaction(['queue'], 'readwrite');
  const store = tx.objectStore('queue');
  
  return store.delete(id);
}

// Notify client of sync results
async function notifyClient(requestId, status, response) {
  const clients = await self.clients.matchAll();
  
  for (const client of clients) {
    client.postMessage({
      type: 'sync-result',
      requestId,
      status,
      data: response ? await response.json() : null
    });
  }
}

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data.type === 'queue-request') {
    addToQueue(event.data.request).then((id) => {
      event.ports[0].postMessage({ success: true, id });
    }).catch((error) => {
      event.ports[0].postMessage({ success: false, error: error.message });
    });
  }
  
  if (event.data.type === 'trigger-sync') {
    self.registration.sync.register('sync-offline-queue');
  }
});

// Push notifications
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New update available',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Details',
      },
      {
        action: 'close',
        title: 'Dismiss',
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('PolicyCortex Update', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/dashboard')
    );
  }
});