/**
 * LunaLab — Shared JavaScript Utilities
 *
 * Common helpers used across all frontend pages:
 * API calls, toast notifications, formatting, etc.
 */

const API_BASE = '';  // same origin

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(`${API_BASE}${path}`, opts);

    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
            const err = await res.json();
            detail = err.detail || JSON.stringify(err);
        } catch (_) {}
        throw new Error(detail);
    }

    return res.json();
}

const apiGet    = (path) => api('GET', path);
const apiPost   = (path, body) => api('POST', path, body);
const apiDelete = (path) => api('DELETE', path);

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

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined) return '—';
    const n = Number(value);
    if (Number.isNaN(n)) return '—';

    if (Math.abs(n) >= 1e6) return n.toExponential(decimals);
    if (Math.abs(n) < 0.01 && n !== 0) return n.toExponential(decimals);

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

function formatEnergy(joules) {
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
    paper_bgcolor: '#111827',
    plot_bgcolor:  '#111827',
    font: {
        family: 'Inter, sans-serif',
        color:  '#94a3b8',
        size:   12,
    },
    margin: { t: 40, r: 30, b: 50, l: 60 },
    xaxis: {
        gridcolor:  'rgba(42, 49, 80, 0.5)',
        zerolinecolor: 'rgba(42, 49, 80, 0.8)',
        linecolor: 'rgba(42, 49, 80, 0.8)',
    },
    yaxis: {
        gridcolor:  'rgba(42, 49, 80, 0.5)',
        zerolinecolor: 'rgba(42, 49, 80, 0.8)',
        linecolor: 'rgba(42, 49, 80, 0.8)',
    },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

// Colour palette for comparison charts
const COMPARISON_COLORS = [
    '#6366f1', '#22d3ee', '#34d399', '#fbbf24',
    '#f43f5e', '#a78bfa', '#f97316', '#06b6d4',
];
