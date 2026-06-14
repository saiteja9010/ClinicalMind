'use strict';

/* ─── Language strings ───────────────────────────────────────── */
const LANG = {
  en: {
    tagline:             'Hospital AI Platform',
    verifiedLabel:       'Verified AI',
    trust1:              'HIPAA Compliant',
    trust2:              'Clinically Validated',
    trust3:              '256-bit Encrypted',
    trust4:              'Results in Seconds',
    heroTitle:           'Get Your AI Clinical Summary',
    heroSub:             'Upload a medical image and describe your case — our AI generates a comprehensive clinical summary instantly.',
    uploadLabel:         'Medical Image',
    optionalTag:         'Optional',
    dropTitle:           'Drag & drop image here',
    browseBtn:           'browse files',
    questionLabel:       'Clinical Question',
    questionPlaceholder: 'Describe your symptoms or ask a medical question…\n\nExample: A 45-year-old male presents with chest pain radiating to the left arm. What is the most likely diagnosis?',
    charLabel:           'characters',
    btnCta:              'Get AI Summary',
    ctaNote:             'Your data is processed securely and never stored.',
    loadingTitle:        'Analyzing Your Case',
    loadingSub:          'Our AI is generating your clinical summary…',
    resultTitle:         'AI Clinical Summary',
    copyBtn:             'Copy',
    copied:              'Copied!',
    newQueryBtn:         'New Query',
    dismissBtn:          'Dismiss',
    disclaimer:          'For informational purposes only. Always consult a qualified healthcare professional.',
    footerCopy:          '© 2024 ClinicalMind · AI-powered clinical decision support',
    footerPrivacy:       'Privacy Policy',
    footerTerms:         'Terms of Use',
    footerSupport:       'Support',
    errNoQuestion:       'Please enter your clinical question.',
    msLabel:             'ms',
  },
  hi: {
    tagline:             'Hospital AI Platform',
    verifiedLabel:       'Verified AI',
    trust1:              'HIPAA Compliant',
    trust2:              'Clinically Validated',
    trust3:              '256-bit Encrypted',
    trust4:              'Turant Results',
    heroTitle:           'Apni AI Clinical Summary Paayein',
    heroSub:             'Medical image upload karein aur case describe karein — hamara AI seconds mein comprehensive clinical summary generate karta hai.',
    uploadLabel:         'Medical Image',
    optionalTag:         'Optional',
    dropTitle:           'Yahan image drag & drop karein',
    browseBtn:           'files browse karein',
    questionLabel:       'Clinical Sawaal',
    questionPlaceholder: 'Apne symptoms bataaiye ya medical sawaal poochiye…\n\nUdaaharan: 45 saal ke purush ko left arm tak jaata hua chest pain hai. Sambhavit diagnosis kya hai?',
    charLabel:           'characters',
    btnCta:              'AI Summary Lein',
    ctaNote:             'Aapka data securely process hota hai aur store nahi kiya jaata.',
    loadingTitle:        'Aapka Case Analyze Ho Raha Hai',
    loadingSub:          'Hamara AI aapki clinical summary generate kar raha hai…',
    resultTitle:         'AI Clinical Summary',
    copyBtn:             'Copy Karein',
    copied:              'Copy Ho Gaya!',
    newQueryBtn:         'Naya Sawaal',
    dismissBtn:          'Band Karein',
    disclaimer:          'Sirf jaankari ke liye. Kisi qualified healthcare professional se zaroor milein.',
    footerCopy:          '© 2024 ClinicalMind · AI-powered clinical decision support',
    footerPrivacy:       'Privacy Policy',
    footerTerms:         'Terms of Use',
    footerSupport:       'Support',
    errNoQuestion:       'Kripya apna clinical sawaal darj karein.',
    msLabel:             'ms',
  },
};

/* ─── State ──────────────────────────────────────────────────── */
let currentLang      = 'en';
let currentImageFile = null;

/* ─── DOM refs ───────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const el = {
  langScreen:     $('langScreen'),
  appScreen:      $('appScreen'),
  langSwitchBtn:  $('langSwitchBtn'),
  langSwitchLabel:$('langSwitchLabel'),
  dropZone:       $('dropZone'),
  dropInner:      $('dropInner'),
  dropLink:       $('dropLink'),
  fileInput:      $('fileInput'),
  imagePreview:   $('imagePreview'),
  removeImg:      $('removeImg'),
  questionInput:  $('questionInput'),
  charCount:      $('charCount'),
  analyzeBtn:     $('analyzeBtn'),
  resultsArea:    $('resultsArea'),
  loadingCard:    $('loadingCard'),
  errorCard:      $('errorCard'),
  errorMsg:       $('errorMsg'),
  summaryCard:    $('summaryCard'),
  summaryText:    $('summaryText'),
  timeChip:       $('timeChip'),
  copyBtn:        $('copyBtn'),
  copyLabel:      $('copyLabel'),
};

/* ─── Init (once DOM is ready) ───────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setupDragDrop();
  setupCharCount();
});

/* ─── Language selection ─────────────────────────────────────── */
function selectLang(lang) {
  currentLang = lang;

  el.langSwitchLabel.textContent = lang === 'en' ? 'EN' : 'HI';

  // Fade out language screen
  el.langScreen.classList.add('fade-out');
  setTimeout(() => {
    el.langScreen.classList.add('hidden');
    el.appScreen.classList.remove('hidden');
    applyLang();
  }, 380);
}

function goToLangScreen() {
  el.appScreen.classList.add('hidden');
  el.langScreen.classList.remove('hidden', 'fade-out');
  // Reset results
  el.resultsArea.classList.add('hidden');
}

/* ─── Apply language strings ─────────────────────────────────── */
function applyLang() {
  const t = LANG[currentLang];

  // Update all [data-key] elements
  document.querySelectorAll('[data-key]').forEach(node => {
    const key = node.getAttribute('data-key');
    if (t[key] !== undefined) node.textContent = t[key];
  });

  // Update placeholders
  document.querySelectorAll('[data-placeholder-key]').forEach(node => {
    const key = node.getAttribute('data-placeholder-key');
    if (t[key] !== undefined) node.placeholder = t[key];
  });

  // lang attribute on html
  document.documentElement.lang = currentLang === 'hi' ? 'hi' : 'en';
}

/* ─── Drag & drop setup ──────────────────────────────────────── */
function setupDragDrop() {
  const dz = el.dropZone;

  dz.addEventListener('click', e => {
    if (e.target === el.removeImg || el.removeImg.contains(e.target)) return;
    el.fileInput.click();
  });

  el.dropLink.addEventListener('click', e => {
    e.stopPropagation();
    el.fileInput.click();
  });

  el.fileInput.addEventListener('change', () => {
    if (el.fileInput.files[0]) setImage(el.fileInput.files[0]);
  });

  dz.addEventListener('dragover', e => {
    e.preventDefault();
    dz.classList.add('drag-over');
  });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setImage(file);
  });

  el.removeImg.addEventListener('click', e => {
    e.stopPropagation();
    clearImage();
  });
}

function setImage(file) {
  currentImageFile = file;
  const url = URL.createObjectURL(file);
  el.imagePreview.src = url;
  el.imagePreview.classList.remove('hidden');
  el.dropInner.classList.add('hidden');
  el.removeImg.classList.remove('hidden');
}

function clearImage() {
  currentImageFile = null;
  el.imagePreview.src = '';
  el.imagePreview.classList.add('hidden');
  el.dropInner.classList.remove('hidden');
  el.removeImg.classList.add('hidden');
  el.fileInput.value = '';
}

/* ─── Character counter ──────────────────────────────────────── */
function setupCharCount() {
  el.questionInput.addEventListener('input', () => {
    el.charCount.textContent = el.questionInput.value.length;
  });
}

/* ─── Analyze ────────────────────────────────────────────────── */
async function handleAnalyze() {
  const question = el.questionInput.value.trim();
  if (!question) {
    showError(LANG[currentLang].errNoQuestion);
    return;
  }

  const fd = new FormData();
  fd.append('question', question);
  fd.append('model',   'EISumm');
  fd.append('dataset', 'MMCQS');
  if (currentImageFile) fd.append('image', currentImageFile);

  showLoading();

  try {
    const res  = await fetch('/api/summarize', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);

    hideLoading();
    renderSummary(data);
  } catch (err) {
    hideLoading();
    showError(err.message || 'An unexpected error occurred.');
  }
}

/* ─── Render summary ─────────────────────────────────────────── */
function renderSummary(data) {
  el.summaryText.textContent = data.summary || '(no output)';
  el.timeChip.textContent    = `${data.inference_time_ms} ${LANG[currentLang].msLabel}`;
  el.copyLabel.textContent   = LANG[currentLang].copyBtn;

  el.loadingCard.classList.add('hidden');
  el.errorCard.classList.add('hidden');
  el.summaryCard.classList.remove('hidden');

  el.summaryCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ─── Reset form ─────────────────────────────────────────────── */
function resetForm() {
  el.questionInput.value = '';
  el.charCount.textContent = '0';
  clearImage();
  el.resultsArea.classList.add('hidden');
  el.loadingCard.classList.add('hidden');
  el.errorCard.classList.add('hidden');
  el.summaryCard.classList.add('hidden');
  el.analyzeBtn.disabled = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ─── UI state helpers ───────────────────────────────────────── */
function showLoading() {
  el.resultsArea.classList.remove('hidden');
  el.loadingCard.classList.remove('hidden');
  el.errorCard.classList.add('hidden');
  el.summaryCard.classList.add('hidden');
  el.analyzeBtn.disabled = true;
  el.loadingCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideLoading() {
  el.loadingCard.classList.add('hidden');
  el.analyzeBtn.disabled = false;
}

function showError(msg) {
  el.errorMsg.textContent = msg;
  el.resultsArea.classList.remove('hidden');
  el.errorCard.classList.remove('hidden');
  el.loadingCard.classList.add('hidden');
  el.summaryCard.classList.add('hidden');
}

function hideError() {
  el.errorCard.classList.add('hidden');
}

/* ─── Copy summary ───────────────────────────────────────────── */
function copySummary() {
  navigator.clipboard.writeText(el.summaryText.textContent).then(() => {
    el.copyBtn.classList.add('copied');
    el.copyLabel.textContent = LANG[currentLang].copied;
    setTimeout(() => {
      el.copyBtn.classList.remove('copied');
      el.copyLabel.textContent = LANG[currentLang].copyBtn;
    }, 2000);
  }).catch(() => { /* clipboard unavailable — silent */ });
}
