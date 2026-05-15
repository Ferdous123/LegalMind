/**
 * LegalMind — editor.js
 * Inline field editing and correction submission.
 *
 * Usage — mark any element with these data attributes:
 *   data-editable="true"
 *   data-field-path="parties.defendant"      (dot-notation path within the draft)
 *   data-doc-id="doc_abc123"                 (document ID)
 *   data-draft-type="case_fact_summary"      (draft template type)
 *   data-generated-text="original text"      (original model output for this field)
 *   data-source-chunk="source OCR text"      (optional: raw OCR from which text came)
 *
 * Vanilla JS, ES2020+. No external dependencies.
 * Requires showToast() from app.js to be loaded first.
 */

'use strict';

/* =========================================================
   Constants
   ========================================================= */

const CORRECTIONS_ENDPOINT = '/api/v1/corrections';

// Pencil icon SVG (inline, no external refs)
const EDIT_TRIGGER_ICON = `<svg viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M8.5 1.5l2 2L4 10H2v-2l6.5-6.5z" stroke="currentColor" stroke-width="1.2"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

/* =========================================================
   Field initialisation
   ========================================================= */

/**
 * Initialise all elements marked as editable in the document.
 * Safe to call multiple times — will skip already-initialised fields.
 */
function initEditableFields() {
    document.querySelectorAll('[data-editable="true"]').forEach(initField);
}

/**
 * Attach the edit trigger to a single field element.
 *
 * @param {HTMLElement} element - The field element.
 */
function initField(element) {
    if (element.dataset.editorInit === 'true') return;
    element.dataset.editorInit = 'true';

    // Ensure the element is positioned relatively so the trigger can anchor
    if (getComputedStyle(element).position === 'static') {
        element.classList.add('field-editable');
    } else {
        element.classList.add('field-editable');
    }

    const trigger = buildEditTrigger();

    trigger.addEventListener('click', (evt) => {
        evt.stopPropagation();
        evt.preventDefault();
        enterEditMode(element);
    });

    element.appendChild(trigger);
}

/**
 * Build the small "Edit" button that appears on field hover.
 *
 * @returns {HTMLButtonElement}
 */
function buildEditTrigger() {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'edit-trigger';
    btn.setAttribute('aria-label', 'Edit this field');
    btn.innerHTML = `${EDIT_TRIGGER_ICON}<span>Edit</span>`;
    return btn;
}

/* =========================================================
   Edit mode
   ========================================================= */

/**
 * Replace the field's display content with an inline editing widget
 * (textarea + Save / Cancel buttons).
 *
 * @param {HTMLElement} element - The field element.
 */
function enterEditMode(element) {
    // Prevent double-entry
    if (element.dataset.editing === 'true') return;
    element.dataset.editing = 'true';

    const originalText =
        element.dataset.generatedText ??
        element.dataset.currentText ??
        extractFieldText(element);

    // Snapshot the current rendered content so we can restore on cancel
    const originalInnerHTML = element.innerHTML;

    // Build the editing widget
    const wrapper = document.createElement('div');
    wrapper.className = 'field-editing';

    const textarea = document.createElement('textarea');
    textarea.value = originalText;
    textarea.rows = Math.max(2, Math.ceil(originalText.length / 80));
    textarea.setAttribute('aria-label', 'Edit field value');
    textarea.setAttribute('spellcheck', 'true');

    const actions = document.createElement('div');
    actions.className = 'field-editing-actions';

    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.className = 'btn-save-edit';
    saveBtn.textContent = 'Update';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn-cancel-edit';
    cancelBtn.textContent = 'Cancel';

    actions.append(saveBtn, cancelBtn);
    wrapper.append(textarea, actions);

    // Replace field content with the editing widget
    element.innerHTML = '';
    element.appendChild(wrapper);

    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    // --- Event handlers ---

    const handleSave = async () => {
        const newText = textarea.value.trim();
        if (!newText) {
            textarea.focus();
            return;
        }

        saveBtn.disabled = true;
        cancelBtn.disabled = true;
        saveBtn.textContent = 'Saving...';

        await saveCorrection(element, newText, originalText, originalInnerHTML);
    };

    const handleCancel = () => {
        exitEditMode(element, originalInnerHTML);
    };

    saveBtn.addEventListener('click', handleSave);
    cancelBtn.addEventListener('click', handleCancel);

    // Allow Ctrl+Enter to save, Escape to cancel
    textarea.addEventListener('keydown', (evt) => {
        if (evt.key === 'Enter' && (evt.ctrlKey || evt.metaKey)) {
            evt.preventDefault();
            handleSave();
        } else if (evt.key === 'Escape') {
            evt.preventDefault();
            handleCancel();
        }
    });
}

/**
 * Exit editing mode, restoring prior HTML if needed.
 *
 * @param {HTMLElement} element       - The field element.
 * @param {string}      restoredHtml  - HTML to restore into element.
 */
function exitEditMode(element, restoredHtml) {
    delete element.dataset.editing;
    element.innerHTML = restoredHtml;
}

/**
 * Extract the visible text content of a field, excluding the edit trigger
 * button text so we don't capture "Edit" in the original value.
 *
 * @param {HTMLElement} element
 * @returns {string}
 */
function extractFieldText(element) {
    const clone = element.cloneNode(true);
    clone.querySelectorAll('.edit-trigger').forEach((el) => el.remove());
    return clone.textContent.trim();
}

/* =========================================================
   Correction submission
   ========================================================= */

/**
 * POST the corrected text to the server and update the field's display.
 *
 * @param {HTMLElement} fieldEl       - The field element (currently in edit mode).
 * @param {string}      newText       - The corrected text entered by the user.
 * @param {string}      originalText  - The original generated text.
 * @param {string}      originalHtml  - The field's pre-edit innerHTML (for restore).
 */
async function saveCorrection(fieldEl, newText, originalText, originalHtml) {
    const payload = {
        document_id: fieldEl.dataset.docId ?? null,
        draft_type: fieldEl.dataset.draftType ?? null,
        field_path: fieldEl.dataset.fieldPath ?? null,
        source_ocr_chunk: fieldEl.dataset.sourceChunk ?? '',
        generated_text: originalText,
        edited_text: newText,
        correction_type: detectCorrectionType(originalText, newText),
    };

    try {
        const resp = await fetch(CORRECTIONS_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (resp.ok) {
            // Update the stored generated text to reflect the latest saved value
            fieldEl.dataset.generatedText = newText;
            fieldEl.dataset.currentText = newText;

            // Restore the field to display mode showing the new text
            const displayHtml = buildUpdatedFieldHtml(fieldEl, newText);
            exitEditMode(fieldEl, displayHtml);
            markFieldEdited(fieldEl);

            const toast = (window.LegalMind?.showToast) ?? showToastFallback;
            toast('Correction saved. Future drafts will reflect this change.', 'success');
        } else {
            let serverMessage = null;
            try {
                const body = await resp.json();
                serverMessage = body?.detail ?? body?.message ?? null;
            } catch {
                // Response body not JSON — ignore
            }
            const msg = serverMessage ?? 'Failed to save correction. Please try again.';
            const toast = (window.LegalMind?.showToast) ?? showToastFallback;
            toast(msg, 'error');
            // Restore to original on failure so no data is lost
            exitEditMode(fieldEl, originalHtml);
        }
    } catch (networkErr) {
        console.error('[editor.js] saveCorrection network error:', networkErr);
        const toast = (window.LegalMind?.showToast) ?? showToastFallback;
        toast('Network error — correction could not be submitted.', 'error');
        exitEditMode(fieldEl, originalHtml);
    }
}

/**
 * Build the inner HTML for a field that has just been successfully updated.
 * Re-attaches the edit trigger so the field remains editable.
 *
 * @param {HTMLElement} fieldEl  - The field element.
 * @param {string}      newText  - The corrected text.
 * @returns {string}
 */
function buildUpdatedFieldHtml(fieldEl, newText) {
    const escaped = escapeHtml(newText);
    const trigger = buildEditTrigger();
    return `${escaped}${trigger.outerHTML}`;
}

/* =========================================================
   Heuristic: correction type detection
   ========================================================= */

/**
 * Classify the nature of the correction.
 *
 * @param {string} original - Original generated text.
 * @param {string} edited   - User-edited text.
 * @returns {'omission'|'style'|'error'}
 */
function detectCorrectionType(original, edited) {
    if (!original || original.length < 10) return 'omission';
    if (edited.length > original.length * 1.5) return 'omission';
    if (edited.toLowerCase().trim() === original.toLowerCase().trim()) return 'style';
    return 'error';
}

/* =========================================================
   Post-edit badge
   ========================================================= */

/**
 * Append a small "Edited" badge to a field after a successful correction.
 *
 * @param {HTMLElement} element - The field element.
 */
function markFieldEdited(element) {
    // Remove any existing badge first to avoid duplicates
    element.querySelectorAll('.badge-edited').forEach((el) => el.remove());

    const badge = document.createElement('span');
    badge.className = 'badge-edited';
    badge.setAttribute('title', 'This field has been manually corrected');
    badge.textContent = 'Edited';
    element.appendChild(badge);
}

/* =========================================================
   Fallback toast (used if app.js is not loaded)
   ========================================================= */

/**
 * Minimal fallback toast using a console message.
 * In production, app.js should always be loaded first.
 *
 * @param {string} message
 * @param {'success'|'error'} [type='success']
 */
function showToastFallback(message, type = 'success') {
    if (type === 'error') {
        console.error('[LegalMind]', message);
    } else {
        console.info('[LegalMind]', message);
    }
}

/* =========================================================
   Utility — HTML escape (mirrors app.js version)
   ========================================================= */

/**
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

/* =========================================================
   Bootstrap
   ========================================================= */

document.addEventListener('DOMContentLoaded', initEditableFields);

// Also re-init after HTMX swaps new content into the DOM
document.body.addEventListener('htmx:afterSwap', initEditableFields);

// Expose for manual invocation (e.g., after dynamic content insertion)
window.LegalMind = Object.assign(window.LegalMind ?? {}, {
    initEditableFields,
    initField,
    enterEditMode,
    markFieldEdited,
    detectCorrectionType,
});
