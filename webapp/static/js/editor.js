/**
 * LegalMind — editor.js (WYSIWYG)
 * Click-to-edit-in-place over the rendered-markdown view.
 *
 * Trigger: any element with data-editable="true". Inside it, prefer a
 * .rendered-markdown child as the contenteditable surface (so the edit
 * area excludes the trigger button itself).
 *
 * On save: turn the edited HTML back into markdown via Turndown.js and
 * POST the correction. The HTML view is then kept as-is (no re-render
 * round-trip), so the operator sees exactly what they just typed.
 *
 * Required: marked.js + turndown.js + turndown-plugin-gfm loaded in the
 * page (drafts.html provides both).  Requires showToast() from base.html.
 */

'use strict';

const CORRECTIONS_ENDPOINT = '/api/v1/corrections';

const EDIT_TRIGGER_ICON = `<svg viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M8.5 1.5l2 2L4 10H2v-2l6.5-6.5z" stroke="currentColor" stroke-width="1.2"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

const CHECK_ICON = `<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M3 8.5l3 3 7-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

const X_ICON = `<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>`;

/* =========================================================
   Turndown setup (HTML -> Markdown)
   ========================================================= */

let _turndown = null;
function getTurndown() {
    if (_turndown) return _turndown;
    if (typeof TurndownService === 'undefined') return null;
    _turndown = new TurndownService({
        headingStyle: 'atx',          // # H1
        hr: '---',
        bulletListMarker: '-',
        codeBlockStyle: 'fenced',
        emDelimiter: '*',
    });
    // GFM tables / strikethrough / task lists if the plugin loaded
    if (typeof turndownPluginGfm !== 'undefined') {
        try { _turndown.use(turndownPluginGfm.gfm); } catch(_) {}
    }
    // Keep our citation-ref spans as plain [E1] tokens, not arbitrary HTML
    _turndown.addRule('citationRef', {
        filter: (node) => node.nodeName === 'SPAN' && node.classList.contains('citation-ref'),
        replacement: (content) => content,
    });
    return _turndown;
}

/* =========================================================
   Field initialisation
   ========================================================= */

function initEditableFields() {
    document.querySelectorAll('[data-editable="true"]').forEach(initField);
}

function initField(element) {
    if (element.dataset.editorInit === 'true') return;
    element.dataset.editorInit = 'true';
    element.classList.add('field-editable');

    const trigger = buildEditTrigger();
    trigger.addEventListener('click', (evt) => {
        evt.stopPropagation();
        evt.preventDefault();
        enterEditMode(element);
    });
    element.appendChild(trigger);
}

function buildEditTrigger() {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'edit-trigger';
    btn.setAttribute('aria-label', 'Edit this section');
    btn.innerHTML = `${EDIT_TRIGGER_ICON}<span>Edit</span>`;
    return btn;
}

/* =========================================================
   Edit mode (contenteditable)
   ========================================================= */

function enterEditMode(element) {
    if (element.dataset.editing === 'true') return;
    element.dataset.editing = 'true';

    // Prefer the rendered-markdown surface so the edit area doesn't
    // include the trigger button. Fall back to the element itself.
    const surface = element.querySelector('.rendered-markdown') || element;

    const originalHTML = surface.innerHTML;
    const originalMarkdown = element.dataset.generatedText || '';

    // Hide the trigger while editing (we have the floating toolbar instead)
    const trigger = element.querySelector(':scope > .edit-trigger');
    if (trigger) trigger.style.display = 'none';

    // Strip any prior "Edited" badge so it doesn't get baked into the
    // contenteditable area while typing
    surface.querySelectorAll('.badge-edited').forEach(el => el.remove());

    // Activate contenteditable
    surface.contentEditable = 'true';
    surface.classList.add('editing-surface');
    surface.setAttribute('spellcheck', 'true');

    // Floating toolbar (sticky at top of the editable surface)
    const toolbar = document.createElement('div');
    toolbar.className = 'edit-toolbar';
    toolbar.innerHTML = `
      <span class="edit-toolbar-status">Editing — changes save as a correction</span>
      <button type="button" class="btn-edit-cancel" aria-label="Cancel edit">${X_ICON}<span>Cancel</span></button>
      <button type="button" class="btn-edit-save" aria-label="Save correction">${CHECK_ICON}<span>Save</span></button>
    `;
    element.insertBefore(toolbar, surface);

    const saveBtn = toolbar.querySelector('.btn-edit-save');
    const cancelBtn = toolbar.querySelector('.btn-edit-cancel');
    const statusEl = toolbar.querySelector('.edit-toolbar-status');

    // Focus and place caret at end
    setTimeout(() => {
        surface.focus();
        try {
            const range = document.createRange();
            range.selectNodeContents(surface);
            range.collapse(false);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        } catch(_) {}
    }, 30);

    const cleanup = () => {
        surface.contentEditable = 'false';
        surface.classList.remove('editing-surface');
        surface.removeAttribute('spellcheck');
        toolbar.remove();
        if (trigger) trigger.style.display = '';
        delete element.dataset.editing;
        document.removeEventListener('keydown', onKey);
    };

    const handleSave = async () => {
        const td = getTurndown();
        if (!td) {
            (window.showToast || console.error)('Markdown converter not loaded — refresh the page and retry.', 'error');
            return;
        }

        // Convert the edited HTML back to markdown
        let newMarkdown = '';
        try {
            // Clean up the surface clone before converting so we don't
            // capture the toolbar / trigger if they accidentally got in
            const clone = surface.cloneNode(true);
            clone.querySelectorAll('.edit-trigger, .edit-toolbar, .badge-edited').forEach(n => n.remove());
            newMarkdown = td.turndown(clone.innerHTML).trim();
        } catch (err) {
            console.warn('turndown failed:', err);
            newMarkdown = surface.innerText.trim();
        }

        if (!newMarkdown) {
            statusEl.textContent = 'Cannot save empty content';
            statusEl.style.color = 'rgb(var(--signal-fail))';
            return;
        }
        if (newMarkdown === originalMarkdown.trim()) {
            statusEl.textContent = 'No changes';
            // Bail out without saving
            setTimeout(cleanup, 400);
            return;
        }

        saveBtn.disabled = true;
        cancelBtn.disabled = true;
        statusEl.textContent = 'Saving correction…';

        const payload = {
            document_id: element.dataset.docId || null,
            draft_type: element.dataset.draftType || null,
            field_path: element.dataset.fieldPath || 'content_markdown',
            source_ocr_chunk: element.dataset.sourceChunk || '',
            generated_text: originalMarkdown,
            edited_text: newMarkdown,
            correction_type: detectCorrectionType(originalMarkdown, newMarkdown),
        };

        try {
            const resp = await fetch(CORRECTIONS_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                let detail = `HTTP ${resp.status}`;
                try {
                    const body = await resp.json();
                    detail = body?.detail || body?.message || detail;
                } catch(_) {}
                throw new Error(detail);
            }
            // Update the stored "generated" markdown to the new value so
            // subsequent edits compare against the latest saved version
            element.dataset.generatedText = newMarkdown;
            element.dataset.currentText = newMarkdown;

            (window.showToast || console.info)('Correction saved. Future drafts will reflect this change.', 'success');

            // Re-stamp the "Edited" badge on the wrapper
            markFieldEdited(element);
            cleanup();
        } catch (err) {
            console.error('saveCorrection failed:', err);
            saveBtn.disabled = false;
            cancelBtn.disabled = false;
            statusEl.textContent = 'Save failed: ' + err.message;
            statusEl.style.color = 'rgb(var(--signal-fail))';
        }
    };

    const handleCancel = () => {
        surface.innerHTML = originalHTML;
        cleanup();
    };

    const onKey = (evt) => {
        if (evt.key === 'Escape') {
            evt.preventDefault();
            handleCancel();
        } else if (evt.key === 'Enter' && (evt.ctrlKey || evt.metaKey)) {
            evt.preventDefault();
            handleSave();
        }
    };

    saveBtn.addEventListener('click', handleSave);
    cancelBtn.addEventListener('click', handleCancel);
    document.addEventListener('keydown', onKey);
}

/* =========================================================
   Heuristic: correction type detection
   ========================================================= */

function detectCorrectionType(original, edited) {
    if (!original || original.length < 10) return 'omission';
    if (edited.length > original.length * 1.5) return 'omission';
    if (edited.toLowerCase().trim() === original.toLowerCase().trim()) return 'style';
    return 'error';
}

/* =========================================================
   Edited badge
   ========================================================= */

function markFieldEdited(element) {
    element.querySelectorAll('.badge-edited').forEach((el) => el.remove());
    const badge = document.createElement('span');
    badge.className = 'badge-edited';
    badge.setAttribute('title', 'This section has been manually corrected');
    badge.textContent = 'Edited';
    element.appendChild(badge);
}

/* =========================================================
   Bootstrap
   ========================================================= */

document.addEventListener('DOMContentLoaded', initEditableFields);
document.body.addEventListener('htmx:afterSwap', initEditableFields);

window.LegalMind = Object.assign(window.LegalMind || {}, {
    initEditableFields,
    initField,
    enterEditMode,
    markFieldEdited,
    detectCorrectionType,
});
