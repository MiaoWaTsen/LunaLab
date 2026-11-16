/**
 * LunaLab — interface translation.
 *
 * The pages are written in English and stay that way.  This script
 * translates what is on screen by looking each piece of text up in
 * /static/i18n/messages.json, keyed by the English source itself
 * (gettext-style), and keeps watching: text that page scripts insert later
 * is translated as it appears.  Adding a language is a column in that file;
 * a string with no translation simply stays in English.
 *
 * What counts as one piece of text (a "unit"):
 *
 *   An element whose content is text with only inline markup is translated
 *   as a whole, because word order differs between languages.  Its inline
 *   children become numbered placeholders in the key:
 *
 *       <p>The crater fixes the <strong>energy</strong>, not the size.</p>
 *       key: "The crater fixes the <0>energy</0>, not the size."
 *
 *   The translation moves the same <strong> element to wherever <0> sits,
 *   so ids, classes and event listeners survive.  A child that page scripts
 *   address by id, or that holds no words (an icon, a number), is opaque:
 *   written <0/>, moved but never rewritten.
 *
 *   Keys may contain {0}, {1}… for values a script interpolates:
 *       "{0} craters (limit {1})"
 *
 *   Icons, arrows and opaque children at either end are not part of the
 *   key: "<0/> Craters", "📋 Craters" and "Craters →" all look up "Craters".
 *
 * Not translated: charts (Plotly redraws its own labels), anything marked
 * translate="no" or data-no-i18n (user names, raw data), and numbers.
 *
 * Loaded before app.js on every page.  Exposes window.LunaI18n.
 */
(function () {
    'use strict';

    const LANGUAGES = [
        { code: 'en',    name: 'English' },
        { code: 'zh-TW', name: '繁體中文' },
        { code: 'zh-CN', name: '简体中文' },
        { code: 'ja',    name: '日本語' },
        { code: 'de',    name: 'Deutsch' },
        { code: 'fr',    name: 'Français' },
    ];
    const SOURCE_LANGUAGE = 'en';
    // Bump when messages.json changes, so browsers drop their cached copy
    const MESSAGES_VERSION = '1';
    const MESSAGES_URL = `/static/i18n/messages.json?v=${MESSAGES_VERSION}`;
    const LANG_KEY = 'lunalab_lang';
    const CACHE_KEY = 'lunalab_i18n_messages';

    const INLINE_TAGS = new Set(['STRONG', 'EM', 'B', 'I', 'SPAN', 'A', 'CODE', 'SUB', 'SUP',
                                 'SMALL', 'KBD', 'ABBR', 'BR', 'U', 'MARK']);
    const SKIPPED_TAGS = new Set(['SCRIPT', 'STYLE', 'SVG', 'CANVAS', 'TEXTAREA', 'NOSCRIPT', 'PRE']);
    const TRANSLATED_ATTRIBUTES = ['placeholder', 'title', 'aria-label'];
    // Two Latin letters in a row, or any CJK character: there is a word here
    const HAS_WORDS = /[A-Za-zÀ-ÿ]{2}|[぀-ヿ一-鿿]/;

    let language = SOURCE_LANGUAGE;
    let exact = new Map();        // source -> translation, for the current language
    let patterns = [];            // [{ regex, template }] for sources with {n}
    let messages = null;          // the parsed messages.json
    const seen = new Set();       // every source unit met (for tests and tooling)
    const missing = new Set();    // those with no translation in the current language

    const elementState = new WeakMap();    // unit element -> { source, kids, rendered }
    const textState = new WeakMap();       // text node    -> { source, rendered }
    const attributeState = new WeakMap();  // element      -> { attr: { source, rendered } }
    let titleSource = null;

    // -----------------------------------------------------------------------
    // Language choice
    // -----------------------------------------------------------------------

    function supported(code) {
        return LANGUAGES.some(l => l.code === code);
    }

    function fromBrowser() {
        for (const tag of navigator.languages || [navigator.language || '']) {
            const lower = tag.toLowerCase();
            if (lower.startsWith('zh')) {
                // Traditional script regions; everything else Chinese reads Simplified
                return /-(tw|hk|mo|hant)/.test(lower) ? 'zh-TW' : 'zh-CN';
            }
            const base = lower.split('-')[0];
            if (supported(base)) return base;
        }
        return SOURCE_LANGUAGE;
    }

    function initialLanguage() {
        let stored = null;
        try { stored = localStorage.getItem(LANG_KEY); } catch (_) {}
        return supported(stored) ? stored : fromBrowser();
    }

    // -----------------------------------------------------------------------
    // Messages
    // -----------------------------------------------------------------------

    function escapeRegex(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function buildDictionary() {
        exact = new Map();
        patterns = [];
        missing.clear();
        if (!messages) return;
        const column = messages.languages.indexOf(language);
        if (column < 0) return;
        for (const [source, row] of Object.entries(messages.messages)) {
            const translation = row[column];
            if (translation == null) continue;
            if (/\{\d+\}/.test(source)) {
                const regex = new RegExp('^' + escapeRegex(source).replace(/\\\{(\d+)\\\}/g, '(?<p$1>.+?)') + '$');
                patterns.push({ regex, template: translation, source });
            } else {
                exact.set(source, translation);
            }
        }
    }

    const SYMBOL = /^[\p{S}\p{Extended_Pictographic}\uFE0F\u200D]$/u;

    /**
     * Split off decoration at both ends: whitespace, opaque placeholders and
     * symbols (icons, arrows, stars).  Returns [prefix, core, suffix].
     */
    function splitAffixes(text) {
        let start = 0;
        let end = text.length;
        for (;;) {
            const rest = text.slice(start, end);
            const lead = rest.match(/^(?:\s+|<\d+\/>)/);
            if (lead) { start += lead[0].length; continue; }
            const first = String.fromCodePoint(rest.codePointAt(0) ?? 32);
            // "<" opens a placeholder that has content: part of the core
            if (rest && first !== '<' && SYMBOL.test(first)) { start += first.length; continue; }
            break;
        }
        for (;;) {
            const rest = text.slice(start, end);
            const tail = rest.match(/(?:\s+|<\d+\/>)$/);
            if (tail) { end -= tail[0].length; continue; }
            const chars = Array.from(rest);
            const last = chars[chars.length - 1];
            if (last && last !== '>' && SYMBOL.test(last)) { end -= last.length; continue; }
            break;
        }
        return [text.slice(0, start), text.slice(start, end), text.slice(end)];
    }

    function lookupExactOrPattern(source) {
        const direct = exact.get(source);
        if (direct !== undefined) return direct;
        for (const { regex, template } of patterns) {
            const match = regex.exec(source);
            if (match) {
                // An interpolated value may itself be a phrase we know ("Preset loaded: {0}")
                return template.replace(/\{(\d+)\}/g, (_, n) => {
                    const value = match.groups['p' + n] ?? '';
                    return lookup(value) ?? value;
                });
            }
        }
        return null;
    }

    /** Translation of a source unit in the current language, or null. */
    function lookup(source) {
        const whole = lookupExactOrPattern(source);
        if (whole !== null) return whole;
        const [prefix, core, suffix] = splitAffixes(source);
        if (core === source || !HAS_WORDS.test(core)) return null;
        const inner = lookupExactOrPattern(core);
        return inner === null ? null : prefix + inner + suffix;
    }

    function translate(source) {
        const core = splitAffixes(source)[1];
        if (!HAS_WORDS.test(core)) return source;
        seen.add(core);
        const found = lookup(source);
        if (found !== null) return found;
        // The source language needs no entry: only other languages can lack one
        if (language !== SOURCE_LANGUAGE) missing.add(core);
        return source;
    }

    /** For page scripts: t('Job {0} queued', id). Falls back to the English text. */
    function t(source, ...values) {
        const template = exact.get(source) ?? source;
        seen.add(source);
        if (language !== SOURCE_LANGUAGE && !exact.has(source) && messages) missing.add(source);
        return template.replace(/\{(\d+)\}/g, (_, n) => values[Number(n)] ?? '');
    }

    function loadCached() {
        try {
            const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null');
            if (cached && cached.version === MESSAGES_VERSION) return cached.data;
        } catch (_) {}
        return null;
    }

    async function fetchMessages() {
        const response = await fetch(MESSAGES_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        try {
            localStorage.setItem(CACHE_KEY, JSON.stringify({ version: MESSAGES_VERSION, data }));
        } catch (_) {}   // storage full or unavailable: fetch again next time
        return data;
    }

    // -----------------------------------------------------------------------
    // Reading and writing units
    // -----------------------------------------------------------------------

    function collapse(text) {
        return text.replace(/\s+/g, ' ');
    }

    function isSkipped(element) {
        return SKIPPED_TAGS.has(element.tagName.toUpperCase())
            || element.getAttribute('translate') === 'no'
            || element.hasAttribute('data-no-i18n')
            || element.classList.contains('js-plotly-plot');
    }

    function isOpaque(child) {
        return child.tagName === 'BR' || Boolean(child.id) || isSkipped(child)
            || !HAS_WORDS.test(child.textContent);
    }

    function isInlineSubtree(element) {
        if (!INLINE_TAGS.has(element.tagName)) return false;
        if (isOpaque(element)) return true;
        return Array.from(element.children).every(isInlineSubtree);
    }

    /** An element with words of its own and nothing but inline markup inside. */
    function isUnit(element) {
        let ownWords = false;
        for (const node of element.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                if (HAS_WORDS.test(node.nodeValue)) ownWords = true;
            } else if (node.nodeType === Node.ELEMENT_NODE && !isInlineSubtree(node)) {
                return false;
            }
        }
        return ownWords;
    }

    /**
     * The key for an element's content.  `kids` lists its placeholder
     * elements; pass the list from an earlier call to keep their numbers
     * (a translation may have reordered them).
     */
    function keyOf(element, kids = []) {
        const write = node => {
            let out = '';
            for (const child of node.childNodes) {
                if (child.nodeType === Node.TEXT_NODE) {
                    out += child.nodeValue.replace(/</g, '&lt;');
                } else if (child.nodeType === Node.ELEMENT_NODE) {
                    let index = kids.indexOf(child);
                    if (index < 0) index = kids.push(child) - 1;
                    out += isOpaque(child) ? `<${index}/>` : `<${index}>${write(child)}</${index}>`;
                }
            }
            return out;
        };
        return { key: collapse(write(element)).trim(), kids };
    }

    function placeholders(template) {
        return (template.match(/<(\d+)\/?>/g) || []).map(tag => tag.replace(/\D/g, '')).sort().join(',');
    }

    /** Rebuild an element's content from a template, reusing its child elements. */
    function render(element, kids, template) {
        const root = document.createDocumentFragment();
        const stack = [root];
        const token = /<(\d+)\/>|<(\d+)>|<\/(\d+)>|[^<]+/g;
        let match;
        while ((match = token.exec(template)) !== null) {
            const parent = stack[stack.length - 1];
            if (match[1] !== undefined) {
                parent.appendChild(kids[Number(match[1])]);
            } else if (match[2] !== undefined) {
                const kid = kids[Number(match[2])];
                kid.replaceChildren();
                parent.appendChild(kid);
                stack.push(kid);
            } else if (match[3] !== undefined) {
                stack.pop();
            } else {
                parent.appendChild(document.createTextNode(match[0].replace(/&lt;/g, '<')));
            }
        }
        element.replaceChildren(root);
    }

    function translateUnit(element) {
        let state = elementState.get(element);
        let current = keyOf(element, state ? state.kids.slice() : []);
        if (!state || state.rendered !== current.key) {
            // New text (first visit, or a page script rewrote it): a new source
            current = keyOf(element);
            state = { source: current.key, kids: current.kids, rendered: current.key };
        }
        let target = translate(state.source);
        if (placeholders(target) !== placeholders(state.source)) {
            console.warn('i18n: placeholders differ, left untranslated:', state.source);
            target = state.source;
        }
        if (target !== state.rendered) {
            render(element, state.kids, target);
            state.rendered = keyOf(element, state.kids.slice()).key;
        }
        elementState.set(element, state);
        // Opaque children were moved, not read: they may hold units of their own
        for (const kid of state.kids) {
            if (kid.isConnected && kid.tagName !== 'BR' && !isSkipped(kid) && HAS_WORDS.test(kid.textContent)
                    && isOpaque(kid)) {
                walk(kid);
            }
        }
    }

    function translateTextNode(node) {
        const raw = node.nodeValue;
        const current = collapse(raw).trim();
        if (!current) return;
        let state = textState.get(node);
        if (!state || state.rendered !== current) {
            if (!HAS_WORDS.test(current)) return;
            state = { source: current, rendered: current };
        }
        const target = translate(state.source);
        if (target !== state.rendered) {
            const lead = raw.match(/^\s*/)[0];
            const trail = raw.match(/\s*$/)[0];
            node.nodeValue = lead + target + trail;
            state.rendered = target;
        }
        textState.set(node, state);
    }

    function translateAttributes(element) {
        let states = attributeState.get(element);
        for (const name of TRANSLATED_ATTRIBUTES) {
            const current = element.getAttribute(name);
            if (current === null) continue;
            let state = states && states[name];
            if (!state || state.rendered !== current) {
                if (!HAS_WORDS.test(current)) continue;
                state = { source: current, rendered: current };
            }
            const target = translate(collapse(state.source).trim());
            if (target !== state.rendered) {
                element.setAttribute(name, target);
                state.rendered = target;
            }
            states = states || {};
            states[name] = state;
        }
        if (states) attributeState.set(element, states);
    }

    function walk(element) {
        if (isSkipped(element)) return;
        translateAttributes(element);
        if (isUnit(element)) {
            translateUnit(element);
            return;
        }
        for (const node of Array.from(element.childNodes)) {
            if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
            else if (node.nodeType === Node.ELEMENT_NODE) walk(node);
        }
    }

    function translateTitle() {
        if (titleSource === null) titleSource = document.title;
        document.title = translate(titleSource);
    }

    function translatePage() {
        document.documentElement.lang = language;
        if (!document.body) return;
        translateTitle();
        walk(document.body);
    }

    // -----------------------------------------------------------------------
    // Keeping up with page scripts
    // -----------------------------------------------------------------------

    const pending = new Set();
    let flushScheduled = false;

    function schedule(node) {
        // Start from the block that owns this text: a change inside a <strong>
        // changes the unit its paragraph forms
        let element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
        while (element && element !== document.body && INLINE_TAGS.has(element.tagName)) {
            element = element.parentElement;
        }
        if (!element) return;
        pending.add(element);
        if (!flushScheduled) {
            flushScheduled = true;
            // A microtask: done before the browser paints the untranslated text
            queueMicrotask(flush);
        }
    }

    function flush() {
        flushScheduled = false;
        const batch = Array.from(pending);
        pending.clear();
        for (const element of batch) {
            // Our own writes come back here too; they match what we rendered
            // and change nothing, which is what ends the loop
            if (element.isConnected) walk(element);
        }
    }

    function observe() {
        new MutationObserver(mutations => {
            for (const mutation of mutations) schedule(mutation.target);
        }).observe(document.body, {
            subtree: true,
            childList: true,
            characterData: true,
            attributes: true,
            attributeFilter: TRANSLATED_ATTRIBUTES,
        });
    }

    // -----------------------------------------------------------------------
    // The language button
    // -----------------------------------------------------------------------

    function mountSelector() {
        if (document.querySelector('.lang-select')) return;
        const label = document.createElement('label');
        label.className = 'lang-select';
        label.setAttribute('data-no-i18n', '');
        label.innerHTML = '<span class="lang-select__icon" aria-hidden="true">🌐</span>';
        const select = document.createElement('select');
        select.className = 'lang-select__menu';
        select.setAttribute('aria-label', 'Language');
        for (const { code, name } of LANGUAGES) {
            const option = document.createElement('option');
            option.value = code;
            option.lang = code;
            option.textContent = name;
            select.appendChild(option);
        }
        select.value = language;
        select.addEventListener('change', () => setLanguage(select.value));
        label.appendChild(select);

        const navbar = document.querySelector('.navbar__inner');
        if (navbar) {
            navbar.appendChild(label);
        } else {
            // Pages without a navbar (the login lobby): a corner of the screen
            label.classList.add('lang-select--floating');
            document.body.appendChild(label);
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    function setLanguage(code) {
        if (!supported(code)) return;
        language = code;
        try { localStorage.setItem(LANG_KEY, code); } catch (_) {}
        buildDictionary();
        translatePage();
        const select = document.querySelector('.lang-select__menu');
        if (select) select.value = code;
        document.dispatchEvent(new CustomEvent('lunalab:languagechange', { detail: { language: code } }));
    }

    language = initialLanguage();
    document.documentElement.lang = language;
    messages = loadCached();
    buildDictionary();

    // Resolves when the messages are in and the page has been translated
    const ready = (async () => {
        if (!messages) {
            try {
                messages = await fetchMessages();
            } catch (error) {
                console.warn('i18n: could not load translations; staying in English.', error);
            }
            buildDictionary();
        }
        translatePage();
    })();

    // This script sits at the end of <body>, so the static page is already
    // parsed: with cached messages it is translated here, before first paint.
    if (messages) translatePage();
    observe();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountSelector);
    } else {
        mountSelector();
    }

    window.LunaI18n = {
        languages: LANGUAGES,
        get language() { return language; },
        setLanguage,
        t,
        ready,
        // For tests and for finding what still needs translating
        seen,
        missing,
    };
})();
