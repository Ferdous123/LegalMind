/**
 * LegalMind — evidence.js
 * Evidence panel interaction and citation highlighting.
 *
 * Citation format in draft text: [E1], [E2], [E12], etc.
 * Evidence panel items must have: data-citation-id="E1"
 * The draft container must have: id="draft-content"
 * The evidence panel must have:  id="evidence-panel"
 *
 * Vanilla JS, ES2020+. No external dependencies.
 */

'use strict';

/* =========================================================
   Constants
   ========================================================= */

const CITATION_REGEX = /\[E(\d+)\]/g;
const CLASS_CITATION_REF = 'citation-ref';
const CLASS_CITATION_ACTIVE = 'citation-active';
const CLASS_EVIDENCE_ITEM = 'evidence-item';
const CLASS_EVIDENCE_ACTIVE = 'evidence-item-active';
const CLASS_EVIDENCE_HIGHLIGHT = 'evidence-highlight';
const CLASS_EVIDENCE_HIGHLIGHT_ACTIVE = 'evidence-active';

/* =========================================================
   Citation annotation
   ========================================================= */

/**
 * Walk the text nodes inside draftContentEl and wrap every [Exx] occurrence
 * in a clickable <span class="citation-ref" data-ref="Exx">[Exx]</span>.
 *
 * Works on text nodes only — avoids re-processing already-wrapped spans.
 *
 * @param {HTMLElement} draftContentEl - The container holding draft text.
 */
function annotateEvidenceCitations(draftContentEl) {
    if (!draftContentEl) return;

    // Collect all text nodes that contain at least one [Exx] pattern
    const textNodes = getTextNodesContainingCitations(draftContentEl);

    textNodes.forEach((textNode) => {
        const fragment = convertCitationsInTextNode(textNode);
        if (fragment) {
            textNode.parentNode.replaceChild(fragment, textNode);
        }
    });
}

/**
 * Return all text nodes within root that match the citation pattern.
 *
 * @param {Node} root
 * @returns {Text[]}
 */
function getTextNodesContainingCitations(root) {
    const results = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            // Skip text inside already-annotated citation spans
            if (node.parentElement?.classList.contains(CLASS_CITATION_REF)) {
                return NodeFilter.FILTER_REJECT;
            }
            return CITATION_REGEX.test(node.nodeValue)
                ? NodeFilter.FILTER_ACCEPT
                : NodeFilter.FILTER_SKIP;
        },
    });

    let node;
    while ((node = walker.nextNode())) {
        results.push(node);
    }
    return results;
}

/**
 * Split a text node on [Exx] markers and return a DocumentFragment where
 * each [Exx] is wrapped in a citation span.
 *
 * @param {Text} textNode
 * @returns {DocumentFragment|null} null if no citations found.
 */
function convertCitationsInTextNode(textNode) {
    const raw = textNode.nodeValue;
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    let match;
    let found = false;

    // Reset regex state (global regex retains lastIndex across calls)
    CITATION_REGEX.lastIndex = 0;

    while ((match = CITATION_REGEX.exec(raw)) !== null) {
        found = true;
        const [fullMatch, digits] = match;
        const refId = `E${digits}`;

        // Preceding plain text
        if (match.index > lastIndex) {
            fragment.appendChild(
                document.createTextNode(raw.slice(lastIndex, match.index))
            );
        }

        // Citation span
        const span = document.createElement('span');
        span.className = CLASS_CITATION_REF;
        span.dataset.ref = refId;
        span.setAttribute('role', 'button');
        span.setAttribute('tabindex', '0');
        span.setAttribute('aria-label', `View evidence ${refId}`);
        span.textContent = fullMatch;
        fragment.appendChild(span);

        lastIndex = match.index + fullMatch.length;
    }

    if (!found) return null;

    // Remaining plain text after last match
    if (lastIndex < raw.length) {
        fragment.appendChild(document.createTextNode(raw.slice(lastIndex)));
    }

    return fragment;
}

/* =========================================================
   Click handler — citation ref in draft
   ========================================================= */

/**
 * Handle click (and keyboard activation) on a [Exx] span inside the draft.
 *
 * @param {MouseEvent|KeyboardEvent} evt
 */
function handleCitationClick(evt) {
    const target = evt.target;

    // Keyboard: only act on Enter/Space
    if (evt.type === 'keydown' && evt.key !== 'Enter' && evt.key !== ' ') return;

    if (!target.classList.contains(CLASS_CITATION_REF)) return;

    evt.preventDefault();

    const refId = target.dataset.ref;
    if (!refId) return;

    clearHighlights();
    activateCitationRef(refId);
    scrollEvidencePanelItemIntoView(refId);
}

/**
 * Highlight all [Exx] citation spans in the draft that match refId,
 * and mark the corresponding evidence panel item as active.
 *
 * @param {string} refId - e.g. "E3"
 */
function activateCitationRef(refId) {
    // Highlight all matching spans in draft
    document
        .querySelectorAll(`.${CLASS_CITATION_REF}[data-ref="${CSS.escape(refId)}"]`)
        .forEach((span) => {
            span.classList.add(CLASS_CITATION_ACTIVE);
        });

    // Activate matching evidence panel item
    const panelItem = document.querySelector(
        `#evidence-panel [data-citation-id="${CSS.escape(refId)}"]`
    );
    if (panelItem) {
        panelItem.classList.add(CLASS_EVIDENCE_ACTIVE);
    }

    // Highlight the surrounding sentence/paragraph in the draft if a highlight
    // wrapper exists (optional enhancement — works if the template wraps
    // paragraphs in elements with data-citation-scope="Exx")
    document
        .querySelectorAll(`[data-citation-scope="${CSS.escape(refId)}"]`)
        .forEach((el) => {
            el.classList.add(CLASS_EVIDENCE_HIGHLIGHT_ACTIVE);
        });
}

/**
 * Scroll the corresponding evidence panel item into view.
 *
 * @param {string} refId
 */
function scrollEvidencePanelItemIntoView(refId) {
    const panelItem = document.querySelector(
        `#evidence-panel [data-citation-id="${CSS.escape(refId)}"]`
    );
    if (!panelItem) return;

    panelItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* =========================================================
   Click handler — evidence panel item
   ========================================================= */

/**
 * Handle click on an item inside the evidence panel.
 * Highlights corresponding citation refs in the draft.
 *
 * @param {MouseEvent} evt
 */
function handleEvidencePanelClick(evt) {
    // Traverse up from the click target to find the evidence-item
    const item = evt.target.closest(`.${CLASS_EVIDENCE_ITEM}`);
    if (!item) return;

    const refId = item.dataset.citationId;
    if (!refId) return;

    clearHighlights();

    // Mark this panel item active
    item.classList.add(CLASS_EVIDENCE_ACTIVE);

    // Highlight all matching citation refs in the draft
    highlightDraftCitationsForRef(refId);

    // Scroll first matching citation into view in the draft
    scrollDraftCitationIntoView(refId);
}

/**
 * Add highlight classes to all [Exx] spans in the draft for a given refId.
 *
 * @param {string} refId
 */
function highlightDraftCitationsForRef(refId) {
    document
        .querySelectorAll(`.${CLASS_CITATION_REF}[data-ref="${CSS.escape(refId)}"]`)
        .forEach((span) => {
            span.classList.add(CLASS_CITATION_ACTIVE);
        });

    document
        .querySelectorAll(`[data-citation-scope="${CSS.escape(refId)}"]`)
        .forEach((el) => {
            el.classList.add(CLASS_EVIDENCE_HIGHLIGHT_ACTIVE);
        });
}

/**
 * Scroll the first occurrence of a citation ref in the draft into view.
 *
 * @param {string} refId
 */
function scrollDraftCitationIntoView(refId) {
    const firstRef = document.querySelector(
        `#draft-content .${CLASS_CITATION_REF}[data-ref="${CSS.escape(refId)}"]`
    );
    if (!firstRef) return;
    firstRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* =========================================================
   Clear all highlights
   ========================================================= */

/**
 * Remove all active citation and evidence highlight classes from the document.
 */
function clearHighlights() {
    document
        .querySelectorAll(`.${CLASS_CITATION_REF}.${CLASS_CITATION_ACTIVE}`)
        .forEach((el) => el.classList.remove(CLASS_CITATION_ACTIVE));

    document
        .querySelectorAll(`.${CLASS_EVIDENCE_ITEM}.${CLASS_EVIDENCE_ACTIVE}`)
        .forEach((el) => el.classList.remove(CLASS_EVIDENCE_ACTIVE));

    document
        .querySelectorAll(`.${CLASS_EVIDENCE_HIGHLIGHT_ACTIVE}`)
        .forEach((el) => el.classList.remove(CLASS_EVIDENCE_HIGHLIGHT_ACTIVE));
}

/* =========================================================
   Bootstrap
   ========================================================= */

document.addEventListener('DOMContentLoaded', function () {
    const draftContent = document.getElementById('draft-content');

    if (draftContent) {
        annotateEvidenceCitations(draftContent);

        draftContent.addEventListener('click', handleCitationClick);

        // Keyboard navigation for citation refs (accessibility)
        draftContent.addEventListener('keydown', handleCitationClick);
    }

    const evidencePanel = document.getElementById('evidence-panel');

    if (evidencePanel) {
        evidencePanel.addEventListener('click', handleEvidencePanelClick);
    }

    // Clear highlights when clicking anywhere in the document that is not
    // a citation ref or an evidence panel item
    document.addEventListener('click', (evt) => {
        const clickedCitation = evt.target.closest(`.${CLASS_CITATION_REF}`);
        const clickedEvidenceItem = evt.target.closest(`.${CLASS_EVIDENCE_ITEM}`);
        if (!clickedCitation && !clickedEvidenceItem) {
            clearHighlights();
        }
    });
});

// Re-annotate after HTMX swaps new draft content
document.body.addEventListener('htmx:afterSwap', function (evt) {
    const draftContent = evt.detail?.target?.id === 'draft-content'
        ? evt.detail.target
        : evt.detail?.target?.querySelector?.('#draft-content');

    if (draftContent) {
        clearHighlights();
        annotateEvidenceCitations(draftContent);
    }
});

// Expose for manual invocation
window.LegalMind = Object.assign(window.LegalMind ?? {}, {
    annotateEvidenceCitations,
    clearHighlights,
    activateCitationRef,
});
