/* Conversation records - front end.
 *
 * Two views: upload/list, and the review screen. The review screen keeps the
 * audio, the transcript and the form in step: every drafted answer carries the
 * segments it came from, and clicking one seeks the recording to that moment.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  templates: [],
  config: null,
  conversations: [],
  current: null,
  poll: null,
  saveTimers: new Map(),
  citedSegments: new Set(),
};

/* ---------------------------------------------------------------- helpers */

function fmtTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '--:--';
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
           : `${m}:${String(sec).padStart(2, '0')}`;
}

function fmtDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
}

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  // Attributes are optional: el('div', childA, childB) is as valid as
  // el('div', { class: 'x' }, childA).
  if (attrs && (attrs.nodeType || Array.isArray(attrs) || typeof attrs !== 'object')) {
    children.unshift(attrs);
    attrs = {};
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2).toLowerCase(), v);
    // A textarea's content is its child text, not a value attribute, so set
    // value as a property for every form control.
    else if (k === 'value' && 'value' in node) node.value = v;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); if (body.detail) detail = body.detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

const STATUS_TEXT = {
  uploaded: 'Queued', transcribing: 'Transcribing', aligning: 'Aligning to the agenda',
  drafting: 'Filling in the form', ready: 'Ready', failed: 'Failed',
};

function statusPill(status, detail) {
  const working = ['uploaded', 'transcribing', 'aligning', 'drafting'].includes(status);
  const cls = status === 'failed' ? 'failed' : working ? 'working' : '';
  return el('span', { class: `pill ${cls}`, title: detail || '' },
    working ? el('i', { class: 'dot pulse' }) : null,
    STATUS_TEXT[status] || status);
}

/* ------------------------------------------------------------ home screen */

function showView(name) {
  $('#view-home').classList.toggle('hidden', name !== 'home');
  $('#view-review').classList.toggle('hidden', name !== 'review');
  window.scrollTo({ top: 0 });
}

function renderTemplateOptions() {
  const select = $('#template-id');
  select.replaceChildren(...state.templates.map(t =>
    el('option', { value: t.id }, `${t.name}`)));
  renderTemplatePreview();
}

function renderTemplatePreview() {
  const template = state.templates.find(t => t.id === $('#template-id').value);
  const card = $('#template-preview');
  if (!template) { card.replaceChildren(); return; }
  card.replaceChildren(
    el('div', { class: 'card-head' },
      el('h2', { style: 'font-size:17px', text: 'What gets filled in' }),
      el('p', { style: 'margin:4px 0 0;color:var(--ink-3);font-size:13.5px', text: template.description })),
    el('div', { class: 'card-body', style: 'padding-top:8px' },
      ...template.sections.map(section => el('div', { style: 'padding:10px 0;border-top:1px solid var(--line)' },
        el('div', { style: 'display:flex;gap:8px;align-items:baseline' },
          el('strong', { style: 'font-size:14px', text: section.title }),
          el('span', { style: 'font-size:12.5px;color:var(--ink-3)',
            text: section.kind === 'agenda' ? `· ${section.minutes} min` : '· whole conversation' })),
        el('div', { style: 'font-size:12.5px;color:var(--ink-2);margin-top:2px', text: section.prompt }),
        el('div', { style: 'display:flex;flex-wrap:wrap;gap:5px;margin-top:7px' },
          ...section.fields.map(f => el('span', { class: 'pill', text: f.label })))))));
}

function renderConversationList() {
  const list = $('#conv-list');
  if (!state.conversations.length) {
    list.replaceChildren(el('div', { class: 'empty', text: 'Nothing here yet. Upload a recording to get started.' }));
    return;
  }
  list.replaceChildren(...state.conversations.map(c =>
    el('button', { class: 'conv-row', onclick: () => openConversation(c.id) },
      el('div', { class: 'grow' },
        el('div', { class: 'who', text: c.title || c.report_name || 'Conversation' }),
        el('div', { class: 'meta' },
          [c.report_name, fmtDate(c.occurred_on || c.created_at), fmtTime(c.duration_seconds)]
            .filter(Boolean).join(' · '))),
      c.edited_field_count ? el('span', { class: 'pill edited', text: `${c.edited_field_count} edited` }) : null,
      statusPill(c.status, c.status_detail))));
}

async function refreshConversations() {
  state.conversations = await api('/api/conversations');
  renderConversationList();
  // Keep polling while anything is still being processed.
  const busy = state.conversations.some(c => !['ready', 'failed'].includes(c.status));
  if (busy && !$('#view-home').classList.contains('hidden')) {
    clearTimeout(state.homePoll);
    state.homePoll = setTimeout(refreshConversations, 2500);
  }
}

/* ---------------------------------------------------------------- upload */

/* Two ways in. Where storage can take the bytes itself we hand them straight
 * to it, because a serverless host caps what it will accept in a request body
 * far below the size of a real recording. Otherwise the file comes through the
 * application, which is simpler and fine for a self-hosted deployment. */

async function uploadThroughServer(file, meta) {
  const form = new FormData();
  form.append('audio', file);
  for (const [key, value] of Object.entries(meta)) form.append(key, String(value));
  return api('/api/conversations', { method: 'POST', body: form });
}

async function uploadDirect(file, meta) {
  const ticketForm = new FormData();
  ticketForm.append('filename', file.name);
  ticketForm.append('content_type', file.type || 'application/octet-stream');
  const ticket = await api('/api/uploads', { method: 'POST', body: ticketForm });

  if (!ticket.direct || !ticket.upload_url) return uploadThroughServer(file, meta);

  const put = await fetch(ticket.upload_url, {
    method: ticket.method || 'PUT',
    headers: ticket.headers || {},
    body: file,
  });
  if (!put.ok) {
    throw new Error(
      `Storage rejected the upload (${put.status}). If this is a CORS error, the bucket `
      + 'needs to allow PUT from this origin.');
  }

  return api('/api/conversations/complete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...meta,
      key: ticket.key,
      audio_filename: file.name,
      audio_mime: file.type || '',
    }),
  });
}

function wireUpload() {
  const dropzone = $('#dropzone');
  const input = $('#audio-input');

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('hot'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hot'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('hot');
    if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; showChosenFile(); }
  });
  input.addEventListener('change', showChosenFile);
  $('#template-id').addEventListener('change', renderTemplatePreview);
  $('#occurred-on').value = new Date().toISOString().slice(0, 10);

  function showChosenFile() {
    const file = input.files[0];
    if (!file) return;
    $('#drop-title').textContent = file.name;
    $('#drop-hint').textContent = `${(file.size / 1048576).toFixed(1)} MB · click to choose a different file`;
  }

  $('#upload-form').addEventListener('submit', async e => {
    e.preventDefault();
    const error = $('#upload-error');
    error.classList.add('hidden');

    const file = input.files[0];
    if (!file) { error.textContent = 'Choose an audio file first.'; error.classList.remove('hidden'); return; }
    if (!$('#consent').checked) {
      error.textContent = 'Confirm that both people knew the conversation was being recorded.';
      error.classList.remove('hidden');
      return;
    }

    const meta = {
      title: $('#title').value,
      template_id: $('#template-id').value,
      manager_name: $('#manager-name').value,
      report_name: $('#report-name').value,
      occurred_on: $('#occurred-on').value,
      consent_confirmed: true,
    };

    $('#upload-btn').disabled = true;
    $('#upload-status').innerHTML = '<span class="spin"></span> Uploading…';
    try {
      const created = state.config && state.config.direct_upload
        ? await uploadDirect(file, meta)
        : await uploadThroughServer(file, meta);
      $('#upload-status').textContent = '';
      $('#upload-form').reset();
      $('#drop-title').textContent = 'Drop an audio file here';
      $('#drop-hint').textContent = 'or click to choose · wav, mp3, m4a, flac, ogg, webm';
      $('#occurred-on').value = new Date().toISOString().slice(0, 10);
      openConversation(created.id);
    } catch (err) {
      error.textContent = err.message;
      error.classList.remove('hidden');
      $('#upload-status').textContent = '';
    } finally {
      $('#upload-btn').disabled = false;
    }
  });
}

/* ---------------------------------------------------------------- review */

async function openConversation(id) {
  showView('review');
  clearTimeout(state.homePoll);
  $('#form-sections').replaceChildren(el('div', { class: 'empty' }, el('span', { class: 'spin' }), ' Loading…'));
  await loadConversation(id);
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  const first = !state.current || state.current.id !== id;
  state.current = data;
  renderReview(first);
  loadAudit(id);
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.poll);
  const conversation = state.current;
  if (!conversation || ['ready', 'failed'].includes(conversation.status)) return;
  state.poll = setTimeout(async () => {
    const status = await api(`/api/conversations/${conversation.id}/status`);
    if (!state.current || state.current.id !== conversation.id) return;
    if (status.status !== conversation.status || ['ready', 'failed'].includes(status.status)) {
      await loadConversation(conversation.id);
    } else {
      state.current.status_detail = status.status_detail;
      renderBanner();
      schedulePoll();
    }
  }, 2000);
}

function renderReview(resetAudio) {
  const c = state.current;
  $('#r-title').textContent = c.title || 'Conversation';
  $('#r-sub').textContent = [
    c.manager_name && c.report_name ? `${c.manager_name} → ${c.report_name}` : (c.report_name || ''),
    fmtDate(c.occurred_on || c.created_at),
    c.template_name,
    c.audio_filename,
  ].filter(Boolean).join(' · ');

  const pill = $('#r-source-pill');
  if (c.analysis_provider === 'claude') {
    pill.className = 'pill ai';
    pill.textContent = `drafted by ${c.analysis_model || 'Claude'}`;
  } else if (c.analysis_provider) {
    pill.className = 'pill warn';
    pill.textContent = 'extracted offline';
  } else {
    pill.className = 'pill hidden';
  }

  if (resetAudio) {
    const player = $('#player');
    player.src = `/api/conversations/${c.id}/audio`;
    player.load();
  }

  renderBanner();
  renderStats();
  renderForm();
  renderTranscript();
  renderTiming();
}

function renderBanner() {
  const c = state.current;
  const banner = $('#r-banner');
  banner.replaceChildren();
  if (c.status === 'failed') {
    banner.append(el('div', { class: 'notice err', style: 'margin-top:12px' },
      `Processing failed: ${c.error || c.status_detail || 'unknown error'}`));
  } else if (c.status !== 'ready') {
    banner.append(el('div', { class: 'notice info', style: 'margin-top:12px' },
      el('span', { class: 'spin' }), ' ',
      c.status_detail || STATUS_TEXT[c.status] || c.status));
  } else if (c.metrics && c.metrics.degraded_reason) {
    banner.append(el('div', { class: 'notice warn', style: 'margin-top:12px' },
      `Drafted offline from verbatim excerpts — the Claude step was unavailable (${c.metrics.degraded_reason}). `
      + 'Every entry below is lifted straight from the transcript. Re-run once it is available.'));
  }
}

function renderStats() {
  const c = state.current;
  const m = c.metrics || {};
  const stats = $('#r-stats');
  stats.replaceChildren();
  if (c.status !== 'ready') return;

  const manager = m.talk_ratio && m.talk_ratio.manager;
  const report = m.talk_ratio && m.talk_ratio.report;

  stats.append(
    el('div', { class: 'stat' },
      el('div', { class: 'k', text: 'Length' }),
      el('div', { class: 'v', text: fmtTime(c.duration_seconds) })),
    manager != null ? el('div', { class: 'stat' },
      el('div', { class: 'k', text: 'Talk time' }),
      el('div', { class: 'v', text: `${Math.round(manager * 100)}% manager` }),
      el('div', { class: 'bar' },
        el('i', { class: 'mgr', style: `width:${manager * 100}%` }),
        el('i', { class: 'rep', style: `width:${report * 100}%` }))) : null,
    el('div', { class: 'stat' },
      el('div', { class: 'k', text: 'Questions asked' }),
      el('div', { class: 'v', text: String((m.questions_by_role && m.questions_by_role.manager) ?? '—') })),
    el('div', { class: 'stat' },
      el('div', { class: 'k', text: 'Fields edited' }),
      el('div', { class: 'v', text: `${c.edited_field_count} of ${c.field_count}` })),
  );
}

/* ------------------------------------------------------------- the form */

function sectionTiming(sectionId) {
  const sections = (state.current.metrics && state.current.metrics.sections) || [];
  return sections.find(s => s.section_id === sectionId);
}

function renderForm() {
  const c = state.current;
  const host = $('#form-sections');
  host.replaceChildren();

  if (c.status !== 'ready' && !c.fields.length) {
    host.append(el('div', { class: 'empty' }, el('span', { class: 'spin' }), ' Working through the recording…'));
    return;
  }

  for (const section of c.template.sections) {
    const fields = c.fields.filter(f => f.section_id === section.id);
    if (!fields.length) continue;

    const timing = sectionTiming(section.id);
    const head = el('div', { class: 'section-head' },
      el('h3', { text: section.title }),
      section.kind === 'agenda' ? el('span', { class: 'mins', text: `· ${section.minutes} min` }) : null,
      timing && timing.segment_count
        ? el('button', {
            class: 'actual', style: 'border:0;background:none;cursor:pointer',
            title: 'Jump to this part of the recording',
            onclick: () => seekTo(timing.start),
          }, `${fmtTime(timing.start)}–${fmtTime(timing.end)} in the recording`)
        : (section.kind === 'agenda' ? el('span', { class: 'actual', text: 'not detected' }) : null));

    host.append(el('div', { class: 'section' },
      head,
      el('div', { class: 'section-prompt', text: section.prompt }),
      ...fields.map(renderField)));
  }
}

function fieldHead(field) {
  const tools = el('div', { class: 'tools' });
  if (field.edited) {
    tools.append(el('button', {
      class: 'btn tiny ghost', text: 'Restore draft',
      title: 'Put the automatically drafted value back',
      onclick: () => revertField(field),
    }));
  }
  if (field.kind === 'list' || field.kind === 'actions') {
    tools.append(el('button', { class: 'btn tiny', text: '+ Add', onclick: () => addItem(field) }));
  }
  return el('div', { class: 'formfield-head' },
    el('label', { class: 'lbl', text: field.label }),
    field.edited
      ? el('span', { class: 'pill edited', text: 'edited by manager' })
      : confidencePill(field),
    tools);
}

function renderField(field) {
  const wrapper = el('div', { class: `formfield ${field.edited ? 'is-edited' : ''}`, 'data-field': field.id });

  wrapper.append(fieldHead(field));

  if (field.guidance) wrapper.append(el('div', { class: 'guidance', text: field.guidance }));
  wrapper.append(renderInput(field));

  if (field.evidence && field.evidence.length) {
    wrapper.append(el('div', { class: 'evidence' },
      ...field.evidence.map(e => el('button', {
        class: 'cite', title: `Jump to ${fmtTime(e.start)} in the recording`,
        onmouseenter: () => highlightSegments([e.segment_index]),
        onmouseleave: () => highlightSegments([]),
        onclick: () => { seekTo(e.start); scrollToSegment(e.segment_index); },
      }, el('time', { text: fmtTime(e.start) }), el('q', { text: e.quote })))));
  }
  return wrapper;
}

function confidencePill(field) {
  if (!field.source || field.source === 'manager') return null;
  const label = field.source === 'claude' ? 'drafted' : 'extracted';
  const confidence = Math.round((field.confidence || 0) * 100);
  return el('span', {
    class: field.source === 'claude' ? 'pill ai' : 'pill warn',
    title: 'Automatically filled in from the recording. Edit freely — your version becomes the record.',
  }, `${label} · ${confidence}% confident`);
}

function renderInput(field) {
  const value = field.value;

  if (field.kind === 'actions') {
    const items = Array.isArray(value) ? value : [];
    const host = el('div');
    if (!items.length) host.append(el('div', { class: 'guidance', text: 'No commitments were recorded.' }));
    items.forEach((action, i) => {
      host.append(el('div', { class: 'action-item' },
        el('textarea', {
          rows: 2, value: action.action || '', placeholder: 'What was agreed',
          oninput: e => updateActionField(field, i, 'action', e.target.value),
        }),
        el('div', { class: 'row' },
          labelled('Owner', el('input', {
            type: 'text', value: action.owner || '',
            oninput: e => updateActionField(field, i, 'owner', e.target.value) })),
          labelled('Due', el('input', {
            type: 'text', value: action.due || '',
            oninput: e => updateActionField(field, i, 'due', e.target.value) })),
          labelled('Support', el('input', {
            type: 'text', value: action.support || '',
            oninput: e => updateActionField(field, i, 'support', e.target.value) }))),
        el('div', { style: 'margin-top:6px;text-align:right' },
          el('button', { class: 'btn tiny ghost', text: 'Remove', onclick: () => removeItem(field, i) }))));
    });
    return host;
  }

  if (field.kind === 'list') {
    const items = Array.isArray(value) ? value : [];
    const host = el('div');
    if (!items.length) host.append(el('div', { class: 'guidance', text: 'Nothing was recorded here.' }));
    items.forEach((item, i) => {
      host.append(el('div', { class: 'list-item' },
        el('textarea', {
          rows: 1, value: item,
          oninput: e => { autoGrow(e.target); updateListItem(field, i, e.target.value); },
        }),
        el('button', { class: 'btn tiny ghost', text: '×', title: 'Remove', onclick: () => removeItem(field, i) })));
    });
    queueMicrotask(() => $$('textarea', host).forEach(autoGrow));
    return host;
  }

  if (field.kind === 'choice') {
    return el('select', { onchange: e => saveField(field, e.target.value) },
      el('option', { value: '' }, '—'),
      ...field.choices.map(choice =>
        el('option', { value: choice, selected: choice === value }, choice)));
  }

  const textarea = el('textarea', {
    rows: 3, value: value || '', placeholder: field.placeholder || '',
    oninput: e => { autoGrow(e.target); saveField(field, e.target.value, true); },
  });
  queueMicrotask(() => autoGrow(textarea));
  return textarea;
}

function labelled(label, input) {
  return el('div', el('div', { class: 'mini-lbl', text: label }), input);
}

function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.max(textarea.scrollHeight, 38)}px`;
}

/* ------------------------------------------------------------- mutations */

function localField(fieldId) {
  return state.current.fields.find(f => f.id === fieldId);
}

function updateListItem(field, index, value) {
  const local = localField(field.id);
  const items = Array.isArray(local.value) ? [...local.value] : [];
  items[index] = value;
  saveField(field, items, true);
}

function updateActionField(field, index, key, value) {
  const local = localField(field.id);
  const items = Array.isArray(local.value) ? local.value.map(a => ({ ...a })) : [];
  items[index] = { ...(items[index] || {}), [key]: value };
  saveField(field, items, true);
}

function addItem(field) {
  const local = localField(field.id);
  const items = Array.isArray(local.value) ? [...local.value] : [];
  items.push(field.kind === 'actions' ? { action: '', owner: '', due: '', support: '' } : '');
  saveField(field, items, false);
}

function removeItem(field, index) {
  const local = localField(field.id);
  const items = Array.isArray(local.value) ? [...local.value] : [];
  items.splice(index, 1);
  saveField(field, items, false);
}

function saveField(field, value, debounce = false) {
  const local = localField(field.id);
  local.value = value;
  local.edited = JSON.stringify(value) !== JSON.stringify(local.draft_value);

  clearTimeout(state.saveTimers.get(field.id));
  const commit = async () => {
    $('#r-save-state').textContent = 'Saving…';
    try {
      const saved = await api(`/api/conversations/${state.current.id}/fields/${field.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: local.value, edited_by: state.current.manager_name || '' }),
      });
      Object.assign(local, saved);
      $('#r-save-state').textContent = 'Saved';
      setTimeout(() => { if ($('#r-save-state').textContent === 'Saved') $('#r-save-state').textContent = ''; }, 1600);
      refreshEditedCount();
      if (!debounce) renderForm();
      else markEdited(field.id, local.edited);
      loadAudit(state.current.id);
    } catch (err) {
      $('#r-save-state').textContent = `Not saved: ${err.message}`;
    }
  };
  state.saveTimers.set(field.id, setTimeout(commit, debounce ? 700 : 0));
}

function markEdited(fieldId, edited) {
  const node = $(`.formfield[data-field="${fieldId}"]`);
  if (!node) return;
  node.classList.toggle('is-edited', edited);
  // Swap the badge and tools in place; the input keeps focus and the cursor.
  const head = $('.formfield-head', node);
  const field = localField(fieldId);
  if (head && field) head.replaceWith(fieldHead(field));
}

async function revertField(field) {
  const saved = await api(`/api/conversations/${state.current.id}/fields/${field.id}/revert`, { method: 'POST' });
  Object.assign(localField(field.id), saved);
  refreshEditedCount();
  renderForm();
  loadAudit(state.current.id);
}

function refreshEditedCount() {
  state.current.edited_field_count = state.current.fields.filter(f => f.edited).length;
  renderStats();
}

/* ----------------------------------------------------------- transcript */

function renderTranscript() {
  const c = state.current;
  const host = $('#transcript');
  host.replaceChildren();

  if (!c.segments.length) {
    host.append(el('div', { class: 'empty' },
      c.status === 'failed' ? 'No transcript was produced.' : 'The transcript will appear here.'));
    return;
  }

  const titleOf = id => {
    const section = c.template.sections.find(s => s.id === id);
    return section ? section.title : (id || 'Unassigned');
  };

  let current = Symbol('none');
  for (const segment of c.segments) {
    if (segment.section_id !== current) {
      current = segment.section_id;
      host.append(el('div', { class: 't-section', text: titleOf(current) }));
    }
    const who = segment.speaker_role === 'manager' ? (c.manager_name || 'Manager')
              : segment.speaker_role === 'report' ? (c.report_name || 'Person coached')
              : (segment.speaker_label || 'Speaker');
    host.append(el('button', {
      class: 't-line', 'data-index': segment.index,
      onclick: () => seekTo(segment.start),
    },
      el('time', { text: fmtTime(segment.start) }),
      el('div',
        el('span', { class: `who ${segment.speaker_role}`, text: who }),
        el('p', { text: segment.text }))));
  }
}

function highlightSegments(indices) {
  state.citedSegments = new Set(indices);
  $$('.t-line').forEach(line => {
    line.classList.toggle('cited', state.citedSegments.has(Number(line.dataset.index)));
  });
}

function scrollToSegment(index) {
  const line = $(`.t-line[data-index="${index}"]`);
  if (line) line.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function seekTo(seconds) {
  const player = $('#player');
  if (!player.src) return;
  player.currentTime = Math.max(0, seconds - 0.4);
  player.play().catch(() => { /* autoplay may be blocked; the seek still landed */ });
}

function followPlayhead() {
  const c = state.current;
  if (!c || !c.segments.length) return;
  const at = $('#player').currentTime;
  const active = c.segments.find(s => at >= s.start && at < s.end);
  $$('.t-line').forEach(line => {
    line.classList.toggle('active', !!active && Number(line.dataset.index) === active.index);
  });
}

function renderTiming() {
  const c = state.current;
  const host = $('#timing');
  const sections = (c.metrics && c.metrics.sections) || [];
  host.replaceChildren();
  if (!sections.length) { host.append(el('div', { class: 'empty', text: 'No timing yet.' })); return; }

  const longest = Math.max(...sections.map(s => Math.max(s.planned_seconds, s.actual_seconds)), 1);
  host.append(el('p', { style: 'font-size:12.5px;color:var(--ink-3);margin:0 0 12px' },
    'How long each part actually took, against the plan.'));

  for (const section of sections) {
    const over = section.actual_seconds > section.planned_seconds * 1.25;
    const under = section.segment_count && section.actual_seconds < section.planned_seconds * 0.5;
    host.append(el('div', { style: 'margin-bottom:14px' },
      el('div', { style: 'display:flex;gap:8px;align-items:baseline' },
        el('strong', { style: 'font-size:13.5px', text: section.title }),
        el('span', { style: 'margin-left:auto;font-size:12px;color:var(--ink-3);font-variant-numeric:tabular-nums',
          text: `${fmtTime(section.actual_seconds)} of ${fmtTime(section.planned_seconds)} planned` })),
      el('div', { style: 'height:6px;border-radius:3px;background:var(--line);margin-top:5px;overflow:hidden' },
        el('i', { style: `display:block;height:100%;width:${Math.min(section.actual_seconds / longest * 100, 100)}%;`
          + `background:${over ? '#d9a441' : under ? '#c2c8d0' : 'var(--accent)'}` })),
      !section.segment_count
        ? el('div', { style: 'font-size:12px;color:var(--warn);margin-top:3px', text: 'This part was not detected in the recording.' })
        : over ? el('div', { style: 'font-size:12px;color:var(--ink-3);margin-top:3px', text: 'Ran over the planned time.' })
        : under ? el('div', { style: 'font-size:12px;color:var(--ink-3);margin-top:3px', text: 'Covered in less than half the planned time.' })
        : null));
  }
}

/* ---------------------------------------------------------------- audit */

const AUDIT_TEXT = {
  uploaded: 'Recording uploaded',
  transcribed: 'Transcribed',
  aligned: 'Aligned to the agenda',
  drafted: 'Form filled in',
  field_edited: 'Field edited by the manager',
  field_reverted: 'Field restored to the draft',
  reprocess_requested: 'Re-run requested',
  failed: 'Processing failed',
};

async function loadAudit(id) {
  const events = await api(`/api/conversations/${id}/audit`);
  const host = $('#audit-list');
  host.replaceChildren(el('ul', {},
    ...events.slice().reverse().map(event => {
      const bits = [];
      const detail = event.detail || {};
      if (detail.field_id) bits.push(`${detail.section_id}.${detail.field_id}`);
      if (detail.method) bits.push(detail.method);
      if (detail.segments) bits.push(`${detail.segments} segments`);
      if (detail.degraded_reason) bits.push(detail.degraded_reason);
      if (detail.error) bits.push(detail.error);
      return el('li', {},
        el('time', { text: new Date(event.at).toLocaleString() }),
        el('span', { text: AUDIT_TEXT[event.action] || event.action }),
        bits.length ? el('span', { style: 'color:var(--ink-3)', text: ` — ${bits.join(' · ')}` }) : null,
        event.actor && event.actor !== 'system'
          ? el('span', { style: 'color:var(--ink-3)', text: ` (${event.actor})` }) : null);
    })));
}

/* ------------------------------------------------------------ page wiring */

function wireReview() {
  $('#player').addEventListener('timeupdate', followPlayhead);

  $$('.tab').forEach(tab => tab.addEventListener('click', () => {
    $$('.tab').forEach(t => t.classList.toggle('on', t === tab));
    $('#transcript').classList.toggle('hidden', tab.dataset.tab !== 'transcript');
    $('#timing').classList.toggle('hidden', tab.dataset.tab !== 'timing');
  }));

  $('#btn-export-md').addEventListener('click', () =>
    window.open(`/api/conversations/${state.current.id}/export.md?transcript=true`, '_blank'));
  $('#btn-export-json').addEventListener('click', () =>
    window.open(`/api/conversations/${state.current.id}/export.json`, '_blank'));

  $('#btn-reprocess').addEventListener('click', async () => {
    if (!confirm('Re-run the pipeline? Fields you have edited are kept as they are.')) return;
    await api(`/api/conversations/${state.current.id}/reprocess`, { method: 'POST' });
    await loadConversation(state.current.id);
  });

  $('#btn-delete').addEventListener('click', async () => {
    if (!confirm('Delete this conversation, its recording and its record? This cannot be undone.')) return;
    await api(`/api/conversations/${state.current.id}`, { method: 'DELETE' });
    state.current = null;
    showView('home');
    refreshConversations();
  });

  $('#nav-home').addEventListener('click', () => {
    clearTimeout(state.poll);
    showView('home');
    refreshConversations();
  });
  $('#nav-new').addEventListener('click', () => {
    clearTimeout(state.poll);
    showView('home');
    refreshConversations();
    $('#upload-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

function renderConfigFooter() {
  const config = state.config;
  if (!config) return;
  $('#config-foot').replaceChildren(
    el('span', { text: `Speech to text: ${config.speech_provider} (${config.speech_model})` }),
    el('span', { text: `Alignment & drafting: ${config.analysis_provider} (${config.analysis_model})` }),
    el('span', { text: `Storage: ${config.storage_backend}${config.direct_upload ? ' (direct upload)' : ''}` }),
    el('span', { text: `Upload limit: ${config.max_upload_mb} MB` }),
    el('span', { text: 'Quotes are resolved from the stored transcript, never generated.' }));
}

async function main() {
  wireUpload();
  wireReview();
  const [templates, config] = await Promise.all([api('/api/templates'), api('/api/config')]);
  state.templates = templates;
  state.config = config;
  renderTemplateOptions();
  renderConfigFooter();
  await refreshConversations();
}

main().catch(err => {
  document.body.prepend(el('div', { class: 'notice err', style: 'margin:16px' },
    `Could not start: ${err.message}`));
});
