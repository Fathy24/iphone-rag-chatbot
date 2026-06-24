/*
 * Header-toggled slide-in panels (Live dashboard + Test cases).
 *
 * Chainlit 2.11's native docked ElementSidebar does not reliably mount JSX
 * custom elements, so we render each panel as a normal inline element
 * (Dashboard.jsx -> [data-cl-dashboard], TestCases.jsx -> [data-cl-tests]) and
 * use this script to turn it into an on-demand slide-in drawer.
 *
 * IMPORTANT: we never move an element in the DOM. Moving a React-managed node
 * breaks Chainlit's live prop updates (React reconciles against the original
 * parent and throws, freezing the element). Instead we:
 *
 *   1. Tag the element + hide its origin chat bubble (CSS only), so it leaves
 *      the chat flow but stays exactly where React rendered it -> live updates
 *      keep flowing.
 *   2. Promote the element itself to a fixed, right-docked drawer via CSS.
 *   3. Toggle a `<open>` body class from the matching header link, a floating
 *      close button, or the Escape key. Opening one panel closes the others.
 *
 * Defensive throughout: if anything fails the panels just stay inline (they are
 * only hidden once we've successfully promoted them to a drawer).
 */
(function () {
  "use strict";

  var PANELS = [
    {
      link: "Live dashboard",
      selector: "[data-cl-dashboard]",
      ready: "cl-dash-ready",
      open: "cl-dash-open",
      host: "cl-dash-host",
      closeId: "cl-dash-close",
      closeLabel: "Close dashboard",
    },
    {
      link: "Test cases",
      selector: "[data-cl-tests]",
      ready: "cl-tests-ready",
      open: "cl-tests-open",
      host: "cl-tests-host",
      closeId: "cl-tests-close",
      closeLabel: "Close test cases",
    },
  ];

  function closeAll() {
    for (var i = 0; i < PANELS.length; i++) {
      document.body.classList.remove(PANELS[i].open);
    }
  }

  function ensureCloseButton(panel) {
    var btn = document.getElementById(panel.closeId);
    if (btn) return btn;
    btn = document.createElement("button");
    btn.id = panel.closeId;
    btn.className = "cl-panel-close";
    btn.setAttribute("aria-label", panel.closeLabel);
    btn.textContent = "\u00d7"; // ×
    btn.addEventListener("click", function () {
      document.body.classList.remove(panel.open);
    });
    document.body.appendChild(btn);
    return btn;
  }

  // Find the chat message bubble wrapping the element, so we can hide its chrome
  // (avatar / padding) WITHOUT removing the element from React's tree.
  function findHost(node) {
    var c = node;
    var guard = 0;
    while (
      c.parentElement &&
      c.parentElement !== document.body &&
      c.parentElement.children.length <= 1 &&
      guard < 14
    ) {
      c = c.parentElement;
      guard++;
    }
    if (
      c.parentElement &&
      c.parentElement !== document.body &&
      c.parentElement.children.length <= 4
    ) {
      return c.parentElement;
    }
    return c;
  }

  // Promote an (in-place) panel element to the drawer and hide its bubble.
  function markLive(panel) {
    var el = document.querySelector(panel.selector);
    if (!el) return false;
    document.body.classList.add(panel.ready);
    ensureCloseButton(panel);
    var host = findHost(el);
    if (host && host !== document.body && !host.classList.contains(panel.host)) {
      host.classList.add(panel.host);
    }
    return true;
  }

  // Attach the open/close toggle to the matching header link.
  function wireLinks() {
    var candidates = document.querySelectorAll("a, button");
    for (var i = 0; i < candidates.length; i++) {
      var el = candidates[i];
      if (el.dataset.clPanelWired || el.className === "cl-panel-close") continue;
      var text = (el.textContent || "").trim();
      for (var p = 0; p < PANELS.length; p++) {
        var panel = PANELS[p];
        if (text.indexOf(panel.link) !== -1) {
          el.dataset.clPanelWired = "1";
          (function (pnl) {
            el.addEventListener("click", function (e) {
              e.preventDefault();
              e.stopPropagation();
              markLive(pnl);
              var isOpen = document.body.classList.contains(pnl.open);
              closeAll();
              if (!isOpen) document.body.classList.add(pnl.open);
            });
          })(panel);
          break;
        }
      }
    }
  }

  function tick() {
    try {
      for (var i = 0; i < PANELS.length; i++) markLive(PANELS[i]);
      wireLinks();
    } catch (err) {
      /* never let panel wiring break the app */
    }
  }

  // Chainlit is a single-page app that re-renders constantly, so observe the
  // DOM (to re-hide a freshly re-rendered bubble / re-wire the links) and poll
  // as a safety net.
  if (document.documentElement) {
    new MutationObserver(function () {
      tick();
    }).observe(document.documentElement, { childList: true, subtree: true });
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeAll();
  });
  document.addEventListener("DOMContentLoaded", tick);
  window.addEventListener("load", tick);
  setInterval(tick, 1200);
  tick();
})();
