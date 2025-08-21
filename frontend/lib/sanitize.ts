import DOMPurify from 'isomorphic-dompurify';

// Configure DOMPurify for secure HTML sanitization
const sanitizeConfig = {
  ALLOWED_TAGS: [
    'b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li',
    'blockquote', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'table', 'thead', 'tbody', 'tr', 'th', 'td', 'span', 'div'
  ],
  ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id'],
  ALLOW_DATA_ATTR: false,
  ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
  ADD_TAGS: [],
  ADD_ATTR: [],
  FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form', 'input'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover'],
  FORCE_BODY: true,
  SANITIZE_DOM: true,
  KEEP_CONTENT: true,
  IN_PLACE: false,
  USE_PROFILES: { html: true }
};

/**
 * Sanitize HTML content to prevent XSS attacks
 * @param dirty - The potentially unsafe HTML string
 * @param options - Optional DOMPurify configuration overrides
 * @returns Sanitized HTML string safe for rendering
 */
export function sanitizeHTML(dirty: string, options?: any): string {
  if (!dirty) return '';
  
  const config = options ? { ...sanitizeConfig, ...options } : sanitizeConfig;
  const clean = DOMPurify.sanitize(dirty, config);
  
  // Convert to string if needed
  const cleanString = typeof clean === 'string' ? clean : clean.toString();
  
  // Additional security: ensure all links open in new tab with proper rel attributes
  if (typeof window !== 'undefined') {
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = cleanString;
    
    const links = tempDiv.getElementsByTagName('a');
    for (let i = 0; i < links.length; i++) {
      const link = links[i];
      if (link.href && !link.href.startsWith(window.location.origin)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    }
    
    return tempDiv.innerHTML;
  }
  
  return cleanString;
}

/**
 * Sanitize markdown-rendered HTML (more permissive for code blocks)
 * @param markdown - The markdown-rendered HTML
 * @returns Sanitized HTML safe for rendering
 */
export function sanitizeMarkdown(markdown: string): string {
  return sanitizeHTML(markdown, {
    ADD_TAGS: ['hr', 'details', 'summary', 'mark'],
    ADD_ATTR: ['data-language', 'data-line'],
  });
}

/**
 * Convert plain text to safe HTML with basic formatting
 * @param text - Plain text to convert
 * @returns Safe HTML string
 */
export function textToSafeHTML(text: string): string {
  if (!text) return '';
  
  // Escape HTML entities
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
  
  // Convert newlines to <br> tags
  const withBreaks = escaped.replace(/\n/g, '<br />');
  
  // Convert **bold** to <strong>
  const withBold = withBreaks.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Convert *italic* to <em>
  const withItalic = withBold.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  return withItalic;
}

/**
 * Strip all HTML tags from a string
 * @param html - HTML string to strip
 * @returns Plain text without HTML tags
 */
export function stripHTML(html: string): string {
  if (!html) return '';
  return DOMPurify.sanitize(html, { ALLOWED_TAGS: [], KEEP_CONTENT: true });
}