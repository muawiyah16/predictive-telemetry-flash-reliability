// SSD Analyzer – main.js
document.addEventListener('DOMContentLoaded', () => {
  // Auto-dismiss flash messages after 5 seconds
  setTimeout(() => {
    document.querySelectorAll('.flash').forEach(el => {
      el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      el.style.opacity = '0';
      el.style.transform = 'translateX(20px)';
      setTimeout(() => el.remove(), 400);
    });
  }, 5000);

  // Animate stat card numbers
  document.querySelectorAll('.sc-val').forEach(el => {
    const raw = el.textContent.trim();
    const num = parseFloat(raw.replace(/[^0-9.]/g, ''));
    if (!isNaN(num) && num > 0) {
      const suffix = raw.replace(/[0-9.]/g, '');
      let start = 0;
      const duration = 1200;
      const step = num / (duration / 16);
      const iv = setInterval(() => {
        start = Math.min(start + step, num);
        el.textContent = (Number.isInteger(num) ? Math.round(start) : start.toFixed(1)) + suffix;
        if (start >= num) clearInterval(iv);
      }, 16);
    }
  });

  // Table row highlight pulse on hover
  document.querySelectorAll('.data-table tbody tr').forEach(row => {
    row.style.transition = 'background 0.15s ease';
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
    });
  });
});
