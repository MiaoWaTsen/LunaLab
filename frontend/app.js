/**
 * LunaLab — Shared JavaScript Utilities
 *
 * Common helpers used across all frontend pages:
 * API calls, toast notifications, formatting, etc.
 */

const API_BASE = '';  // same origin

// ---------------------------------------------------------------------------
// Authentication & Token helpers
// ---------------------------------------------------------------------------

function getAuthToken() {
    return localStorage.getItem('lunalab_token');
}

function setAuthToken(token) {
    if (token) {
        localStorage.setItem('lunalab_token', token);
    } else {
        localStorage.removeItem('lunalab_token');
    }
}

function getStoredUser() {
    try {
        const u = localStorage.getItem('lunalab_user');
        return u ? JSON.parse(u) : null;
    } catch (_) {
        return null;
    }
}

function setStoredUser(user) {
    if (user) {
        localStorage.setItem('lunalab_user', JSON.stringify(user));
    } else {
        localStorage.removeItem('lunalab_user');
    }
}

// ---------------------------------------------------------------------------
// Login guard — every page except /login requires a session
// ---------------------------------------------------------------------------

const LOGIN_PATH = '/login';

function isLoginPage() {
    return window.location.pathname === LOGIN_PATH;
}

function redirectToLogin() {
    if (isLoginPage()) return;
    const next = window.location.pathname + window.location.search;
    window.location.replace(`${LOGIN_PATH}?next=${encodeURIComponent(next)}`);
}

// Only allow same-site relative paths as the post-login destination
function getPostLoginTarget() {
    const next = new URLSearchParams(window.location.search).get('next');
    if (next && next.startsWith('/') && !next.startsWith('//') && next !== LOGIN_PATH) {
        return next;
    }
    return '/';
}

if (!isLoginPage() && !getAuthToken()) {
    redirectToLogin();
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

// A thin bar at the top of the page while requests are in flight.  It waits
// a moment first, so quick requests (most of them) show nothing at all.
const NET_PROGRESS_DELAY_MS = 150;
let netRequests = 0;
let netTimer = null;

function netProgressBar() {
    let bar = document.getElementById('net-progress');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'net-progress';
        bar.className = 'net-progress';
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-label', 'Loading');
        document.body.appendChild(bar);
    }
    return bar;
}

function netStart() {
    if (++netRequests === 1) {
        netTimer = setTimeout(() => netProgressBar().classList.add('net-progress--active'), NET_PROGRESS_DELAY_MS);
    }
}

function netDone() {
    if (--netRequests === 0) {
        clearTimeout(netTimer);
        document.getElementById('net-progress')?.classList.remove('net-progress--active');
    }
}

// quiet: no activity bar, for background polling the user did not ask for
async function api(method, path, body = null, { quiet = false } = {}) {
    if (quiet) return apiRequest(method, path, body);
    netStart();
    try {
        return await apiRequest(method, path, body);
    } finally {
        netDone();
    }
}

async function apiRequest(method, path, body) {
    const headers = { 'Content-Type': 'application/json' };
    const token = getAuthToken();
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const opts = { method, headers };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(`${API_BASE}${path}`, opts);

    // Session expired or revoked → back to the login lobby
    // (login/register return 401 for bad credentials, so skip those)
    if (res.status === 401 && !path.startsWith('/api/auth/login') && !path.startsWith('/api/auth/register')) {
        setAuthToken(null);
        setStoredUser(null);
        redirectToLogin();
    }

    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
            const err = await res.json();
            if (Array.isArray(err.detail)) {
                // FastAPI validation errors: [{loc: [..., field], msg}, ...]
                detail = err.detail.map(d => `${(d.loc || []).slice(-1)[0] ?? 'input'}: ${d.msg}`).join('; ');
            } else {
                detail = err.detail || JSON.stringify(err);
            }
        } catch (_) {}
        throw new Error(detail);
    }

    return res.json();
}

const apiGet    = (path, opts) => api('GET', path, null, opts);
const apiPost   = (path, body, opts) => api('POST', path, body, opts);
const apiDelete = (path, opts) => api('DELETE', path, null, opts);


// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------

function ensureToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    return container;
}

function showToast(message, type = 'success', duration = 3500) {
    const container = ensureToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

const _SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

// 3.23e+13 → "3.23 × 10¹³"
function formatScientific(n, decimals = 2) {
    const [mantissa, exp] = n.toExponential(decimals).split('e');
    const power = String(parseInt(exp, 10)).replace(/./g, c => _SUPERSCRIPT[c] ?? c);
    return `${mantissa} × 10${power}`;
}

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined) return '—';
    const n = Number(value);
    if (Number.isNaN(n)) return '—';

    if (Math.abs(n) >= 1e6) return formatScientific(n, decimals);
    if (Math.abs(n) < 0.01 && n !== 0) return formatScientific(n, decimals);

    return n.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: decimals,
    });
}

function formatDate(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    return d.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

// Lengths in metres → "305 m" / "86.4 km"
function formatLength(metres) {
    if (metres === null || metres === undefined || Number.isNaN(Number(metres))) return '—';
    const m = Number(metres);
    if (Math.abs(m) >= 10_000) return formatNumber(m / 1000, 1) + ' km';
    if (Math.abs(m) >= 100) return formatNumber(m, 0) + ' m';
    return formatNumber(m, 2) + ' m';
}

function formatEnergy(joules) {
    if (joules >= 1e21) return formatScientific(joules) + ' J';
    if (joules >= 1e18) return (joules / 1e18).toFixed(2) + ' EJ';
    if (joules >= 1e15) return (joules / 1e15).toFixed(2) + ' PJ';
    if (joules >= 1e12) return (joules / 1e12).toFixed(2) + ' TJ';
    if (joules >= 1e9)  return (joules / 1e9).toFixed(2) + ' GJ';
    if (joules >= 1e6)  return (joules / 1e6).toFixed(2) + ' MJ';
    if (joules >= 1e3)  return (joules / 1e3).toFixed(2) + ' kJ';
    return joules.toFixed(2) + ' J';
}

// ---------------------------------------------------------------------------
// Plotly dark theme layout
// ---------------------------------------------------------------------------

const PLOTLY_LAYOUT_DEFAULTS = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        color:  '#8b95a7',
        size:   12,
    },
    margin: { t: 40, r: 30, b: 50, l: 60 },
    xaxis: {
        gridcolor:  'rgba(148, 163, 184, 0.08)',
        zerolinecolor: 'rgba(148, 163, 184, 0.2)',
        linecolor: 'rgba(148, 163, 184, 0.15)',
    },
    yaxis: {
        gridcolor:  'rgba(148, 163, 184, 0.08)',
        zerolinecolor: 'rgba(148, 163, 184, 0.2)',
        linecolor: 'rgba(148, 163, 184, 0.15)',
    },
};

/**
 * Lunar terrain colour scale anchored so that elevation 0 (undisturbed
 * surface) is always neutral regolith grey: cool shadowed tones below,
 * warm sunlit tones above.
 */
function terrainColorscale(field) {
    const zmin = Math.min(field.z_min, -1e-9);
    const zmax = Math.max(field.z_max, 1e-9);
    const zero = -zmin / (zmax - zmin);
    return [
        [0,                          '#16213a'],
        [zero * 0.45,                '#2c4670'],
        [zero * 0.85,                '#6b7f99'],
        [zero,                       '#a3a9b2'],
        [zero + (1 - zero) * 0.35,   '#cfc3a6'],
        [1,                          '#f3e2b3'],
    ];
}

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

// Crater morphology classes returned by the API
const CRATER_TYPE_LABELS = {
    simple:  'Simple crater',
    complex: 'Complex crater',
    basin:   'Impact basin',
};
const CRATER_TYPE_HINTS = {
    simple:  'Bowl-shaped crater, depth ≈ 1/5 of diameter (lunar craters < ~15 km).',
    complex: 'Walls collapsed into a flat floor with a central peak (lunar craters > ~15 km).',
    basin:   'Multi-ring basin with an inner peak ring (lunar craters > ~300 km).',
};

// Colour palette for comparison charts
const COMPARISON_COLORS = [
    '#6366f1', '#22d3ee', '#34d399', '#fbbf24',
    '#f43f5e', '#a78bfa', '#f97316', '#06b6d4',
];

// ---------------------------------------------------------------------------
// Scientific Web Audio Engine (No external sound files required)
// ---------------------------------------------------------------------------

let _audioCtx = null;
let _soundEnabled = localStorage.getItem('lunalab_sound') !== 'false';

function isSoundEnabled() {
    return _soundEnabled;
}

function setSoundEnabled(enabled) {
    _soundEnabled = !!enabled;
    localStorage.setItem('lunalab_sound', _soundEnabled ? 'true' : 'false');
}

function playImpactSound(energyJoules = 1e12) {
    if (!_soundEnabled) return;
    try {
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtxClass) return;
        if (!_audioCtx) _audioCtx = new AudioCtxClass();
        if (_audioCtx.state === 'suspended') _audioCtx.resume();

        const now = _audioCtx.currentTime;
        const dur = Math.min(2.8, Math.max(1.0, Math.log10(energyJoules + 1) * 0.15));

        // 1. Sub-bass seismic oscillator
        const osc = _audioCtx.createOscillator();
        const oscGain = _audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(140, now);
        osc.frequency.exponentialRampToValueAtTime(32, now + dur);

        oscGain.gain.setValueAtTime(0.4, now);
        oscGain.gain.exponentialRampToValueAtTime(0.001, now + dur);

        osc.connect(oscGain);
        oscGain.connect(_audioCtx.destination);
        osc.start(now);
        osc.stop(now + dur);

        // 2. White noise cratering blast burst
        const bufferSize = _audioCtx.sampleRate * 0.6;
        const buffer = _audioCtx.createBuffer(1, bufferSize, _audioCtx.sampleRate);
        const output = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }

        const whiteNoise = _audioCtx.createBufferSource();
        whiteNoise.buffer = buffer;

        const filter = _audioCtx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(800, now);
        filter.frequency.exponentialRampToValueAtTime(100, now + 0.6);

        const noiseGain = _audioCtx.createGain();
        noiseGain.gain.setValueAtTime(0.35, now);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);

        whiteNoise.connect(filter);
        filter.connect(noiseGain);
        noiseGain.connect(_audioCtx.destination);

        whiteNoise.start(now);
        whiteNoise.stop(now + 0.6);
    } catch (e) {
        console.warn('Audio playback not allowed or failed:', e);
    }
}

function triggerScreenShake() {
    document.body.classList.add('screen-shake');
    setTimeout(() => document.body.classList.remove('screen-shake'), 450);
}

// ---------------------------------------------------------------------------
// Auth Modal & Navbar Status Widget
// ---------------------------------------------------------------------------

function openAuthModal(initialTab = 'login') {
    let modal = document.getElementById('auth-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'auth-modal';
        modal.className = 'modal-backdrop';
        modal.innerHTML = `
            <div class="modal-card">
                <div class="modal-header">
                    <div class="modal-title">
                        Commander Terminal
                    </div>
                    <button class="modal-close" onclick="closeAuthModal()">✕</button>
                </div>
                <div class="auth-tabs">
                    <button class="auth-tab ${initialTab === 'login' ? 'active' : ''}" id="tab-btn-login" onclick="switchAuthTab('login')">Sign In</button>
                    <button class="auth-tab ${initialTab === 'register' ? 'active' : ''}" id="tab-btn-register" onclick="switchAuthTab('register')">Register New Commander</button>
                </div>
                <form id="auth-form" onsubmit="handleAuthSubmit(event)">
                    <div class="form-group" style="margin-bottom: var(--space-md);">
                        <label class="form-label" for="auth-username">Commander Call-sign (Username)</label>
                        <input type="text" id="auth-username" class="form-input" placeholder="e.g. ApolloCommander" required minlength="2" maxlength="30">
                    </div>
                    <div class="form-group" style="margin-bottom: var(--space-lg);">
                        <label class="form-label" for="auth-password">Security Password</label>
                        <input type="password" id="auth-password" class="form-input" placeholder="••••••••" required minlength="4">
                    </div>
                    <div id="auth-error" class="form-error hidden" style="margin-bottom: var(--space-md);"></div>
                    <button type="submit" class="btn btn--primary" id="auth-submit-btn" style="width: 100%;">
                        ${initialTab === 'login' ? 'Sign In' : 'Create Account'}
                    </button>
                </form>
            </div>
        `;
        document.body.appendChild(modal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeAuthModal();
        });
    } else {
        modal.classList.remove('hidden');
    }
    switchAuthTab(initialTab);
    document.getElementById('auth-username').focus();
}

function closeAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) modal.classList.add('hidden');
}

let _currentAuthTab = 'login';
function switchAuthTab(tab) {
    _currentAuthTab = tab;
    const btnLogin = document.getElementById('tab-btn-login');
    const btnReg = document.getElementById('tab-btn-register');
    const submitBtn = document.getElementById('auth-submit-btn');
    const err = document.getElementById('auth-error');
    if (err) err.classList.add('hidden');

    if (tab === 'login') {
        if (btnLogin) btnLogin.classList.add('active');
        if (btnReg) btnReg.classList.remove('active');
        if (submitBtn) submitBtn.textContent = 'Sign In';
    } else {
        if (btnReg) btnReg.classList.add('active');
        if (btnLogin) btnLogin.classList.remove('active');
        if (submitBtn) submitBtn.textContent = 'Create Account';
    }
}

async function handleAuthSubmit(e) {
    e.preventDefault();
    const username = document.getElementById('auth-username').value.trim();
    const password = document.getElementById('auth-password').value;
    const errEl = document.getElementById('auth-error');
    const btn = document.getElementById('auth-submit-btn');

    errEl.classList.add('hidden');
    btn.disabled = true;
    btn.textContent = 'Authenticating…';

    try {
        const endpoint = _currentAuthTab === 'login' ? '/api/auth/login' : '/api/auth/register';
        const res = await apiPost(endpoint, { username, password });

        setAuthToken(res.token);
        setStoredUser(res.user);
        if (isLoginPage()) {
            window.location.replace(getPostLoginTarget());
            return;
        }
        closeAuthModal();
        updateNavUser();
        showToast(`Welcome aboard, Commander ${res.user.username}!`, 'success');
        
        // Trigger event so open pages can refresh data if needed
        window.dispatchEvent(new CustomEvent('auth-changed', { detail: res.user }));
    } catch (err) {
        errEl.textContent = err.message;
        errEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = _currentAuthTab === 'login' ? 'Sign In' : 'Create Account';
    }
}

async function logoutCommander() {
    try {
        await apiPost('/api/auth/logout', {});
    } catch (_) {}
    setAuthToken(null);
    setStoredUser(null);
    window.location.replace(LOGIN_PATH);
}

function escapeHtml(str) {
    return String(str ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
}

function updateNavUser() {
    const user = getStoredUser();
    const container = document.getElementById('nav-user-slot');
    if (!container) return;

    if (user) {
        container.innerHTML = `
            <div class="user-badge" title="${escapeHtml(user.rank_title)}">
                <span class="user-badge__name">${escapeHtml(user.username)}</span>
                <span class="user-badge__rank">Lv.${user.level}</span>
                <span class="user-badge__xp">${user.xp} XP</span>
                <button type="button" class="btn btn--ghost btn--sm" onclick="logoutCommander()" title="Logout" style="padding: 2px 6px; font-size: 0.75rem; color: var(--text-muted);">
                    ✕
                </button>
            </div>
        `;
    } else {
        container.innerHTML = `
            <button type="button" class="btn btn--secondary btn--sm" onclick="openAuthModal('login')" style="font-size: 0.8rem; padding: 4px 10px;">
                Sign In
            </button>
        `;
    }
}

// On narrow screens the links collapse into a menu under a button that names
// the current page.  The button is added here, so the pages' markup (and the
// no-JavaScript fallback, a scrolling strip) stays as it is.
function setupNavMenu() {
    const nav = document.querySelector('.navbar');
    const links = nav?.querySelector('.navbar__links');
    if (!links) return;
    links.id = links.id || 'navbar-links';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'navbar__toggle';
    toggle.setAttribute('aria-controls', links.id);
    toggle.setAttribute('aria-expanded', 'false');
    const current = links.querySelector('.navbar__link--active')?.textContent.trim() || 'Menu';
    toggle.innerHTML = '<span class="navbar__toggle-bars" aria-hidden="true"></span>'
        + `<span class="navbar__toggle-label">${escapeHtml(current)}</span>`;
    toggle.setAttribute('aria-label', `Menu, current page: ${current}`);
    links.before(toggle);
    nav.classList.add('navbar--js');

    const setOpen = open => {
        nav.classList.toggle('navbar--open', open);
        toggle.setAttribute('aria-expanded', String(open));
    };
    toggle.addEventListener('click', () => setOpen(!nav.classList.contains('navbar--open')));
    document.addEventListener('click', e => {
        if (!nav.contains(e.target)) setOpen(false);
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && nav.classList.contains('navbar--open')) {
            setOpen(false);
            toggle.focus();
        }
    });
}

// Check and sync user status on load
document.addEventListener('DOMContentLoaded', async () => {
    setupNavMenu();

    // Inject nav slot if not already present in .navbar__inner
    const navInner = document.querySelector('.navbar__inner');
    if (navInner && !document.getElementById('nav-user-slot')) {
        const slot = document.createElement('div');
        slot.id = 'nav-user-slot';
        slot.style.display = 'flex';
        slot.style.alignItems = 'center';
        slot.style.gap = '8px';
        slot.style.marginLeft = 'auto';
        navInner.appendChild(slot);
    }

    // Show the signed-in user at once from the last known profile, so the
    // navbar does not change shape when the check below returns
    updateNavUser();

    // Verify token with backend
    const token = getAuthToken();
    if (token) {
        try {
            const me = await apiGet('/api/auth/me');
            setStoredUser(me);
            if (isLoginPage()) {
                window.location.replace(getPostLoginTarget());
                return;
            }
        } catch (_) {
            // A 401 has already cleared the token and redirected (see api()).
            // Anything else — a dropped connection, or this request being
            // aborted because the user navigated away — says nothing about the
            // session, so keep the user signed in.  (Clearing here logged
            // people out whenever they clicked a link while this was in flight.)
        }
    }
    updateNavUser();
});

