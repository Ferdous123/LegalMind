/**
 * LegalMind — app.js
 * Core UI utilities: toasts, dialogs, formatting, HTMX integration,
 * nav active state, upload loading state.
 *
 * Vanilla JS, ES2020+. No external dependencies.
 */

'use strict';

/* =========================================================
   Toast notification system
   ========================================================= */

/**
 * SVG icons keyed by toast type (inline, no external refs).
 * @type {Record<string, string>}
 */
const TOAST_ICONS = {
    success: `<svg class="toast-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
        <path d="M5 8l2 2 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
    error: `<svg class="toast-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
        <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`,
    warning: `<svg class="toast-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M8 2L14.5 13.5H1.5L8 2z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
        <path d="M8 6.5v3M8 11.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`,
    info: `<svg class="toast-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
        <path d="M8 7v5M8 5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>`,
};

/**
 * Retrieve or create the singleton toast container element.
 * @returns {HTMLElement}
 */
function getToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        container.setAttribute('role', 'region');
        container.setAttribute('aria-live', 'polite');
        container.setAttribute('aria-label', 'Notifications');
        document.body.appendChild(container);
    }
    return container;
}

/**
 * Display a toast notification.
 *
 * @param {string} message   - Text to display.
 * @param {'success'|'error'|'warning'|'info'} [type='success'] - Visual style.
 * @param {number} [duration=3500] - Auto-dismiss delay in ms. Pass 0 to disable.
 * @returns {HTMLElement} The toast element.
 */
function showToast(message, type = 'success', duration = 3500) {
    const container = getToastContainer();
    const validTypes = ['success', 'error', 'warning', 'info'];
    const safeType = validTypes.includes(type) ? type : 'info';

    const toast = document.createElement('div');
    toast.className = `toast toast-${safeType}`;
    toast.setAttribute('role', 'alert');

    const iconSvg = TOAST_ICONS[safeType] ?? '';

    toast.innerHTML = `
        ${iconSvg}
        <div class="toast-body">${escapeHtml(message)}</div>
        <button class="toast-close" aria-label="Dismiss notification" type="button">&#x2715;</button>
    `;

    container.appendChild(toast);

    const dismiss = () => {
        if (!toast.parentNode) return;
        toast.classList.add('toast-hiding');
        toast.addEventListener('transitionend', () => toast.remove(), { once: true });
        // Fallback removal in case transition event never fires
        setTimeout(() => toast.remove(), 400);
    };

    toast.querySelector('.toast-close').addEventListener('click', dismiss);

    if (duration > 0) {
        setTimeout(dismiss, duration);
    }

    return toast;
}

/* =========================================================
   Confirmation dialog wrapper
   ========================================================= */

/**
 * Show a native browser confirm dialog.
 * For a custom modal implementation this function can be swapped out
 * without changing call sites.
 *
 * @param {string}   message   - Prompt displayed to the user.
 * @param {Function} onConfirm - Called if the user confirms.
 * @param {Function} [onCancel] - Called if the user cancels. Optional.
 */
function confirmAction(message, onConfirm, onCancel) {
    // Using the native dialog keeps zero external dependencies.
    // Replace the body with a custom modal when design warrants it.
    const confirmed = window.confirm(message);
    if (confirmed) {
        onConfirm();
    } else if (typeof onCancel === 'function') {
        onCancel();
    }
}

/* =========================================================
   Timestamp formatting
   ========================================================= */

/**
 * Format an ISO 8601 timestamp into a human-readable string.
 * Uses the browser's Intl API for locale-aware output.
 *
 * @param {string} isoStr - ISO 8601 date string.
 * @returns {string} Formatted date/time, e.g. "May 12, 2025, 3:41 PM".
 */
function formatTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
        const date = new Date(isoStr);
        if (isNaN(date.getTime())) return isoStr;
        return new Intl.DateTimeFormat(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
        }).format(date);
    } catch {
        return isoStr;
    }
}

/* =========================================================
   Confidence score helpers
   ========================================================= */

/**
 * Format a confidence score (0–1) as a percentage string.
 *
 * @param {number} score - Value between 0 and 1.
 * @returns {string} E.g. "87.3%".
 */
const formatConfidence = (score) => `${(score * 100).toFixed(1)}%`;

/**
 * Return a Tailwind color class appropriate for the given confidence score.
 *
 * @param {number} score - Value between 0 and 1.
 * @returns {string} Tailwind text color class.
 */
function confidenceClass(score) {
    if (score >= 0.8) return 'text-emerald-600';
    if (score >= 0.5) return 'text-amber-600';
    return 'text-red-600';
}

/* =========================================================
   HTMX integration — show toasts on server-triggered events
   ========================================================= */

/**
 * Listen for HTMX afterRequest events.
 * Servers can signal toast messages via response headers:
 *   X-Toast-Message: Operation completed
 *   X-Toast-Type: success   (optional, defaults to 'success')
 *
 * Alternatively, the server may include an HX-Trigger header value
 * encoding a JSON object: {"showToast": {"message": "...", "type": "success"}}
 */
document.body.addEventListener('htmx:afterRequest', function (evt) {
    const xhr = evt.detail?.xhr;
    if (!xhr) return;

    // Handle X-Toast-Message response header
    const toastMessage = xhr.getResponseHeader('X-Toast-Message');
    if (toastMessage) {
        const toastType = xhr.getResponseHeader('X-Toast-Type') || 'success';
        showToast(toastMessage, toastType);
    }

    // Handle HTTP error responses that did not carry a toast header
    if (!toastMessage && xhr.status >= 400) {
        const defaultMsg = xhr.status >= 500
            ? 'A server error occurred. Please try again.'
            : 'The request could not be completed.';
        showToast(defaultMsg, 'error');
    }
});

/**
 * Also handle the HTMX showToast event dispatched via HX-Trigger.
 * Server sends: HX-Trigger: {"showToast": {"message": "Saved", "type": "success"}}
 */
document.body.addEventListener('showToast', function (evt) {
    const { message, type = 'success', duration } = evt.detail ?? {};
    if (message) {
        showToast(message, type, duration);
    }
});

/* =========================================================
   Active navigation link highlighting
   ========================================================= */

document.addEventListener('DOMContentLoaded', function () {
    const currentPath = window.location.pathname;

    document.querySelectorAll('.nav-link[href]').forEach((link) => {
        const href = link.getAttribute('href');
        if (!href || href === '#') return;

        // Exact match OR current path starts with href (for sub-paths),
        // but avoid "/" matching everything by requiring href length > 1.
        const isActive =
            currentPath === href ||
            (href.length > 1 && currentPath.startsWith(href));

        if (isActive) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });
});

/* =========================================================
   Upload form — loading state on submit
   ========================================================= */

document.addEventListener('DOMContentLoaded', function () {
    const uploadForms = document.querySelectorAll(
        'form[data-upload-form], form.upload-form'
    );

    uploadForms.forEach((form) => {
        form.addEventListener('submit', function (evt) {
            // Do not apply loading state if native HTML5 validation fails
            if (!form.checkValidity()) return;

            const submitBtn = form.querySelector(
                'button[type="submit"], input[type="submit"]'
            );
            if (!submitBtn) return;

            // Store original label so it can be restored on error
            const originalText = submitBtn.textContent;
            submitBtn.disabled = true;
            submitBtn.classList.add('btn-loading');
            submitBtn.setAttribute('aria-busy', 'true');

            // Restore button if the page is still active after 60 s
            // (guards against hung requests where no navigation occurred)
            const restoreTimer = setTimeout(() => {
                submitBtn.disabled = false;
                submitBtn.classList.remove('btn-loading');
                submitBtn.textContent = originalText;
                submitBtn.setAttribute('aria-busy', 'false');
            }, 60_000);

            // If the form uses HTMX, restore on HTMX completion
            form.addEventListener(
                'htmx:afterRequest',
                () => {
                    clearTimeout(restoreTimer);
                    submitBtn.disabled = false;
                    submitBtn.classList.remove('btn-loading');
                    submitBtn.setAttribute('aria-busy', 'false');
                },
                { once: true }
            );

            // Clean up timer if normal navigation occurs
            window.addEventListener(
                'beforeunload',
                () => clearTimeout(restoreTimer),
                { once: true }
            );
        });
    });
});

/* =========================================================
   Utility helpers
   ========================================================= */

/**
 * Escape HTML special characters to prevent XSS when injecting
 * user-supplied content into innerHTML.
 *
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
    if (typeof str !== 'string') return String(str ?? '');
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;');
}

/**
 * Debounce a function call.
 *
 * @param {Function} fn
 * @param {number} delay - Milliseconds.
 * @returns {Function}
 */
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/* =========================================================
   SSE pipeline progress listener
   ========================================================= */

/**
 * Connect to the pipeline SSE endpoint and relay progress events.
 *
 * Expected event data: {"stage": "ocr", "pct": 25}
 *
 * Dispatches a custom DOM event "pipelineProgress" on document so that
 * page-specific code can react without coupling to this module.
 *
 * @param {Function} [onProgress] - Optional callback(stage, pct).
 * @param {Function} [onComplete] - Optional callback invoked when pct===100.
 * @returns {EventSource} The EventSource instance (call .close() to stop).
 */
function connectPipelineEvents(onProgress, onComplete) {
    const es = new EventSource('/api/v1/events/pipeline');

    es.onmessage = function (evt) {
        let data;
        try {
            data = JSON.parse(evt.data);
        } catch {
            return;
        }
        const { stage = '', pct = 0 } = data;

        // Dispatch a bubbling custom event so any listener on the page can react
        document.dispatchEvent(
            new CustomEvent('pipelineProgress', { detail: { stage, pct }, bubbles: true })
        );

        if (typeof onProgress === 'function') {
            onProgress(stage, pct);
        }

        if (pct >= 100) {
            es.close();
            if (typeof onComplete === 'function') {
                onComplete();
            }
        }
    };

    es.onerror = function () {
        es.close();
    };

    return es;
}

// Expose utilities on window for use in inline scripts / HTMX handlers
window.LegalMind = Object.assign(window.LegalMind ?? {}, {
    showToast,
    confirmAction,
    formatTimestamp,
    formatConfidence,
    confidenceClass,
    escapeHtml,
    debounce,
    connectPipelineEvents,
});
