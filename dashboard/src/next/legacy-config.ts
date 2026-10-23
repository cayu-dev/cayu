/**
 * Today's unchanged pages build links and update the address with the dashboard's base
 * path (lib/links.ts). The prototype routes in the hash, so point that base path at it:
 * their links become `page.html#/sessions/...` and the new router picks them up. This
 * module must load before anything that reads the dashboard config.
 */
window.__CAYU_DASHBOARD_CONFIG__ = { basePath: `${window.location.pathname}#` }
