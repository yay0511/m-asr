from __future__ import annotations

import argparse
import base64
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
import struct
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .main import read_wav
from .pipeline import StreamingCascadePipeline
from .types import PipelineEvent, TranscriptTurn


LIVE_HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>m_asr 实时话筒识别</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --ink: #1d232b;
      --muted: #667085;
      --panel: #fff;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
      --blue: #155eef;
      --warn: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header { background: #fff; border-bottom: 1px solid var(--line); }
    .wrap { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }
    .top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 0;
    }
    h1 { margin: 0; font-size: 20px; line-height: 1.2; letter-spacing: 0; }
    .nav { display: flex; gap: 10px; align-items: center; }
    .nav a { color: var(--accent); text-decoration: none; font-size: 14px; font-weight: 650; }
    main { padding: 20px 0 32px; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 16px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .panel h2 { margin: 0; padding: 14px 16px; border-bottom: 1px solid var(--line); font-size: 15px; }
    .panel-body { padding: 16px; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button {
      appearance: none;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      height: 40px;
      padding: 0 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: #475467; }
    button.stop { background: var(--danger); }
    button:disabled { background: #98a2b3; cursor: not-allowed; }
    canvas {
      width: 100%;
      height: 104px;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }
    .meter {
      height: 10px;
      background: #e4e7ec;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
    }
    .meter span { display: block; height: 100%; width: 0%; background: var(--accent); }
    .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
    .metric { border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-height: 58px; }
    .metric b { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .metric span { font-size: 15px; font-weight: 750; }
    .notice { margin-top: 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .notice.warn { color: var(--warn); }
    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--line); padding: 10px 12px 0; }
    .tab {
      width: auto;
      height: 34px;
      padding: 0 12px;
      background: transparent;
      color: var(--muted);
      border: 1px solid transparent;
      border-radius: 6px 6px 0 0;
    }
    .tab.active { color: var(--ink); background: #f8fafc; border-color: var(--line); border-bottom-color: #f8fafc; }
    .view { display: none; padding: 14px; background: #f8fafc; min-height: 520px; }
    .view.active { display: block; }
    .turn {
      display: grid;
      grid-template-columns: 140px 120px 1fr;
      gap: 10px;
      align-items: start;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .time { color: var(--muted); font-variant-numeric: tabular-nums; }
    .speaker { color: var(--blue); font-weight: 750; overflow-wrap: anywhere; }
    .text { line-height: 1.45; }
    .empty { color: var(--muted); padding: 36px 0; text-align: center; }
    .error { color: var(--danger); font-weight: 650; white-space: pre-wrap; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; color: #344054; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .top { align-items: flex-start; flex-direction: column; }
      .turn { grid-template-columns: 1fr; gap: 4px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <h1>m_asr 实时话筒识别</h1>
      <div class="nav"><a href="/upload">文件上传</a><span id="status">未连接</span></div>
    </div>
  </header>
  <main class="wrap">
    <div class="grid">
      <section class="panel">
        <h2>实时输入</h2>
        <div class="panel-body">
          <div class="controls">
            <button id="startBtn">开始话筒</button>
            <button id="stopBtn" class="stop" disabled>停止</button>
          </div>
          <canvas id="scope" width="640" height="180"></canvas>
          <div class="meter"><span id="level"></span></div>
          <div class="meta">
            <div class="metric"><b>采样率</b><span id="sampleRate">-</span></div>
            <div class="metric"><b>发送帧</b><span id="frames">0</span></div>
            <div class="metric"><b>ASR</b><span id="asrBackend">-</span></div>
            <div class="metric"><b>运行设备</b><span id="runtimeDevice">-</span></div>
          </div>
          <div class="notice" id="notice">点击开始后，浏览器会请求麦克风权限；音频只发送到本机服务。</div>
        </div>
      </section>
      <section class="panel">
        <div class="tabs">
          <button class="tab active" type="button" data-view="transcript">Transcript</button>
          <button class="tab" type="button" data-view="events">Events</button>
          <button class="tab" type="button" data-view="json">JSON</button>
        </div>
        <div id="transcript" class="view active"><div class="empty">实时识别结果会显示在这里</div></div>
        <div id="events" class="view"><div class="empty">事件流会显示在这里</div></div>
        <div id="json" class="view"><pre>[]</pre></div>
      </section>
    </div>
  </main>
  <script>
    const TARGET_SR = 16000;
    const SEND_MS = 100;
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const scope = document.getElementById('scope');
    const ctx = scope.getContext('2d');
    const levelEl = document.getElementById('level');
    const framesEl = document.getElementById('frames');
    const events = [];
    const turns = new Map();
    let ws = null;
    let audioCtx = null;
    let source = null;
    let processor = null;
    let stream = null;
    let pending = new Float32Array(0);
    let sentFrames = 0;

    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.view).classList.add('active');
      });
    });

    startBtn.addEventListener('click', startLive);
    stopBtn.addEventListener('click', stopLive);

    async function startLive() {
      setStatus('连接中');
      clearViews();
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
      ws.binaryType = 'arraybuffer';
      ws.onmessage = (event) => handleServerMessage(JSON.parse(event.data));
      ws.onerror = () => setError('WebSocket 连接失败');
      ws.onclose = () => setStatus('已断开');
      await waitForOpen(ws);

      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      audioCtx = new AudioContext();
      document.getElementById('sampleRate').textContent = `${audioCtx.sampleRate} -> ${TARGET_SR}`;
      source = audioCtx.createMediaStreamSource(stream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        drawScope(input);
        const resampled = resample(input, audioCtx.sampleRate, TARGET_SR);
        enqueueAndSend(resampled);
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      ws.send(JSON.stringify({ type: 'start', sample_rate: TARGET_SR }));
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus('实时监听中');
    }

    function stopLive() {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (audioCtx) audioCtx.close();
      if (ws && ws.readyState === WebSocket.OPEN) {
        flushPending();
        ws.send(JSON.stringify({ type: 'stop' }));
      }
      startBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus('正在收尾');
    }

    function enqueueAndSend(samples) {
      const joined = new Float32Array(pending.length + samples.length);
      joined.set(pending);
      joined.set(samples, pending.length);
      pending = joined;
      const frameSamples = Math.round(TARGET_SR * SEND_MS / 1000);
      while (pending.length >= frameSamples) {
        const frame = pending.slice(0, frameSamples);
        pending = pending.slice(frameSamples);
        sendInt16(frame);
      }
    }

    function flushPending() {
      if (pending.length) {
        sendInt16(pending);
        pending = new Float32Array(0);
      }
    }

    function sendInt16(samples) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const data = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        data[i] = s < 0 ? s * 32768 : s * 32767;
      }
      ws.send(data.buffer);
      sentFrames += 1;
      framesEl.textContent = sentFrames;
    }

    function handleServerMessage(message) {
      events.push(message);
      document.querySelector('#json pre').textContent = JSON.stringify(events, null, 2);
      if (message.type === 'ready') {
        document.getElementById('asrBackend').textContent = message.backends.asr;
        document.getElementById('runtimeDevice').textContent = `${message.runtime.device} / ${message.runtime.asr_provider}`;
        if (message.warning) {
          const notice = document.getElementById('notice');
          notice.textContent = message.warning;
          notice.className = 'notice warn';
        }
      }
      if (message.type === 'event') {
        renderEvent(message.event);
        if (message.event.event_type === 'partial') {
          const previous = turns.get(message.event.chunk_id) || {};
          turns.set(message.event.chunk_id, { ...previous, ...message.event, speaker_id: previous.speaker_id || '', is_final: false });
          renderTranscript();
        }
        if (message.event.event_type === 'speaker') {
          const previous = turns.get(message.event.chunk_id) || message.event;
          turns.set(message.event.chunk_id, { ...previous, speaker_id: message.event.speaker_id || '', confidence: message.event.confidence });
          renderTranscript();
        }
        if (message.event.event_type === 'final') {
          const previous = turns.get(message.event.chunk_id) || {};
          turns.set(message.event.chunk_id, { ...previous, ...message.event, is_final: true });
          renderTranscript();
        }
      }
      if (message.type === 'stopped') setStatus('已停止');
      if (message.type === 'error') setError(message.error);
    }

    function renderTranscript() {
      const items = mergeTranscriptTurns([...turns.values()].sort((a, b) => a.start - b.start));
      document.getElementById('transcript').innerHTML = items.length ? items.map(renderTurn).join('') : '<div class="empty">暂无最终文本</div>';
    }

    function mergeTranscriptTurns(items) {
      const merged = [];
      for (const turn of items) {
        if (!turn.text) continue;
        const speaker = turn.speaker_id && turn.speaker_id !== 'UNKNOWN' ? turn.speaker_id : '';
        const normalized = { ...turn, speaker_id: speaker };
        const previous = merged[merged.length - 1];
        if (
          previous &&
          previous.speaker_id &&
          normalized.speaker_id &&
          previous.speaker_id === normalized.speaker_id &&
          normalized.start - previous.end <= 1.2
        ) {
          previous.end = Math.max(previous.end, normalized.end);
          previous.text = joinText(previous.text, normalized.text);
          previous.is_final = previous.is_final && normalized.is_final;
          continue;
        }
        merged.push(normalized);
      }
      return merged;
    }

    function joinText(left, right) {
      if (!left) return right || '';
      if (!right) return left || '';
      return `${left}${/[\u3400-\u9fff]$/.test(left) ? '' : ' '}${right}`;
    }

    function renderEvent(event) {
      const container = document.getElementById('events');
      if (container.querySelector('.empty')) container.innerHTML = '';
      const div = document.createElement('div');
      div.className = 'turn';
      div.innerHTML = `
        <div class="time">#${event.chunk_id} ${event.start.toFixed(2)} - ${event.end.toFixed(2)}</div>
        <div class="speaker">${escapeHtml(event.event_type)}</div>
        <div class="text">${escapeHtml(event.speaker_id || '')} ${escapeHtml(event.text || event.message || '')}</div>
      `;
      container.appendChild(div);
    }

    function renderTurn(turn) {
      return `
          <div class="turn">
            <div class="time">${turn.start.toFixed(2)} - ${turn.end.toFixed(2)}</div>
          <div class="speaker">${turn.speaker_id ? escapeHtml(turn.speaker_id) : '<span class="time">...</span>'}</div>
          <div class="text">${escapeHtml(turn.text)}${turn.is_final === false ? '<span class="time"> ...</span>' : ''}</div>
        </div>
      `;
    }

    function resample(input, sourceRate, targetRate) {
      if (sourceRate === targetRate) return new Float32Array(input);
      const ratio = sourceRate / targetRate;
      const outLen = Math.max(1, Math.floor(input.length / ratio));
      const output = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const pos = i * ratio;
        const left = Math.floor(pos);
        const right = Math.min(input.length - 1, left + 1);
        const frac = pos - left;
        output[i] = input[left] * (1 - frac) + input[right] * frac;
      }
      return output;
    }

    function drawScope(samples) {
      ctx.clearRect(0, 0, scope.width, scope.height);
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, scope.width, scope.height);
      ctx.strokeStyle = '#0f766e';
      ctx.beginPath();
      let rms = 0;
      for (let x = 0; x < scope.width; x++) {
        const idx = Math.floor(x * samples.length / scope.width);
        const y = (1 - samples[idx]) * scope.height / 2;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let i = 0; i < samples.length; i++) rms += samples[i] * samples[i];
      rms = Math.sqrt(rms / samples.length);
      levelEl.style.width = `${Math.min(100, rms * 600)}%`;
      ctx.stroke();
    }

    function clearViews() {
      events.length = 0;
      turns.clear();
      sentFrames = 0;
      pending = new Float32Array(0);
      framesEl.textContent = '0';
      document.getElementById('transcript').innerHTML = '<div class="empty">实时识别结果会显示在这里</div>';
      document.getElementById('events').innerHTML = '<div class="empty">事件流会显示在这里</div>';
      document.querySelector('#json pre').textContent = '[]';
    }

    function waitForOpen(socket) {
      return new Promise((resolve, reject) => {
        socket.onopen = resolve;
        socket.onerror = reject;
      });
    }

    function setStatus(text) { statusEl.textContent = text; }
    function setError(text) {
      setStatus('错误');
      document.getElementById('transcript').innerHTML = `<div class="error">${escapeHtml(text)}</div>`;
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[ch]));
    }
  </script>
</body>
</html>
"""


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>m_asr 多说话人流式 ASR</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --ink: #1d232b;
      --muted: #667085;
      --panel: #ffffff;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --warn: #b45309;
      --err: #b42318;
      --speaker: #155eef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .wrap {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }
    .top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 18px 0;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      min-width: 190px;
      text-align: right;
      color: var(--muted);
      font-size: 13px;
    }
    main {
      padding: 20px 0 32px;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      line-height: 1.25;
    }
    .panel-body {
      padding: 16px;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    audio {
      width: 100%;
      margin-top: 12px;
    }
    canvas {
      width: 100%;
      height: 84px;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      height: 40px;
      padding: 0 14px;
      font-weight: 650;
      cursor: pointer;
      width: 100%;
      margin-top: 14px;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      background: #9aa4b2;
      cursor: wait;
    }
    .meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 58px;
    }
    .metric b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 4px;
    }
    .metric span {
      font-size: 15px;
      font-weight: 700;
    }
    .notice {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .notice.warn { color: var(--warn); }
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      padding: 10px 12px 0;
      background: #fff;
    }
    .tab {
      width: auto;
      margin: 0;
      height: 34px;
      padding: 0 12px;
      background: transparent;
      color: var(--muted);
      border-radius: 6px 6px 0 0;
      border: 1px solid transparent;
    }
    .tab.active {
      color: var(--ink);
      background: #f8fafc;
      border-color: var(--line);
      border-bottom-color: #f8fafc;
    }
    .view {
      display: none;
      padding: 14px;
      background: #f8fafc;
      min-height: 420px;
    }
    .view.active { display: block; }
    .turn {
      display: grid;
      grid-template-columns: 140px 120px 1fr;
      gap: 10px;
      align-items: start;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .time { color: var(--muted); font-variant-numeric: tabular-nums; }
    .speaker {
      color: var(--speaker);
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .text { line-height: 1.45; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #344054;
    }
    .empty {
      color: var(--muted);
      padding: 36px 0;
      text-align: center;
    }
    .error {
      color: var(--err);
      font-weight: 650;
      white-space: pre-wrap;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .top { align-items: flex-start; flex-direction: column; }
      .status { text-align: left; }
      .turn { grid-template-columns: 1fr; gap: 4px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <h1>m_asr 多说话人流式 ASR</h1>
      <div class="status" id="status">等待音频</div>
    </div>
  </header>
  <main class="wrap">
    <div class="grid">
      <section class="panel">
        <h2>音频输入</h2>
        <div class="panel-body">
          <form id="form">
            <label for="audioFile">上传 WAV / MP3 / FLAC 音频</label>
            <input id="audioFile" name="audio" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a" required>
            <audio id="player" controls></audio>
            <canvas id="waveform" width="640" height="160"></canvas>
            <button id="submit" type="submit">开始识别</button>
          </form>
          <div class="meta">
            <div class="metric"><b>ASR</b><span id="asrBackend">-</span></div>
            <div class="metric"><b>Speaker</b><span id="speakerBackend">-</span></div>
            <div class="metric"><b>运行设备</b><span id="runtimeDevice">-</span></div>
            <div class="metric"><b>耗时</b><span id="elapsed">-</span></div>
          </div>
          <div class="notice" id="notice">默认优先请求 CUDA；当前环境不可用时自动切换到 CPU。</div>
        </div>
      </section>
      <section class="panel">
        <div class="tabs">
          <button class="tab active" type="button" data-view="transcript">Transcript</button>
          <button class="tab" type="button" data-view="events">Events</button>
          <button class="tab" type="button" data-view="json">JSON</button>
        </div>
        <div id="transcript" class="view active"><div class="empty">识别结果会显示在这里</div></div>
        <div id="events" class="view"><div class="empty">事件流会显示在这里</div></div>
        <div id="json" class="view"><pre>{}</pre></div>
      </section>
    </div>
  </main>
  <script>
    const form = document.getElementById('form');
    const fileInput = document.getElementById('audioFile');
    const submit = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const player = document.getElementById('player');
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');

    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.view).classList.add('active');
      });
    });

    fileInput.addEventListener('change', async () => {
      const file = fileInput.files[0];
      if (!file) return;
      player.src = URL.createObjectURL(file);
      drawWaveform(file);
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;

      setBusy(true);
      setEmpty('transcript', '正在识别...');
      setEmpty('events', '等待事件...');
      document.querySelector('#json pre').textContent = '{}';

      const body = new FormData();
      body.append('audio', file);
      const started = performance.now();

      try {
        const response = await fetch('/api/transcribe', { method: 'POST', body });
        const data = await response.json();
        const elapsed = ((performance.now() - started) / 1000).toFixed(1) + 's';
        document.getElementById('elapsed').textContent = elapsed;
        if (!response.ok) throw new Error(data.error || '识别失败');
        renderResult(data);
        statusEl.textContent = '完成';
      } catch (error) {
        document.getElementById('transcript').innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
        statusEl.textContent = '失败';
      } finally {
        setBusy(false);
      }
    });

    function setBusy(isBusy) {
      submit.disabled = isBusy;
      submit.textContent = isBusy ? '识别中...' : '开始识别';
      statusEl.textContent = isBusy ? '处理中' : statusEl.textContent;
    }

    function renderResult(data) {
      document.getElementById('asrBackend').textContent = data.backends.asr;
      document.getElementById('speakerBackend').textContent = data.backends.speaker;
      document.getElementById('runtimeDevice').textContent = `${data.runtime.device} / ${data.runtime.asr_provider}`;
      document.getElementById('notice').textContent = data.warning || '真实 X-ASR + pyannote 流程已完成。';
      document.getElementById('notice').className = data.warning ? 'notice warn' : 'notice';
      document.getElementById('transcript').innerHTML = renderTurns(data.transcript);
      document.getElementById('events').innerHTML = renderEvents(data.events);
      document.querySelector('#json pre').textContent = JSON.stringify(data, null, 2);
    }

    function renderTurns(turns) {
      const merged = mergeTurns(turns);
      if (!merged.length) return '<div class="empty">没有最终文本</div>';
      return merged.map((turn) => `
        <div class="turn">
          <div class="time">${turn.start.toFixed(2)} - ${turn.end.toFixed(2)}</div>
          <div class="speaker">${turn.speaker_id ? escapeHtml(turn.speaker_id) : '<span class="time">...</span>'}</div>
          <div class="text">${escapeHtml(turn.text)}</div>
        </div>
      `).join('');
    }

    function mergeTurns(turns) {
      const merged = [];
      for (const turn of turns) {
        if (!turn.text) continue;
        const speaker = turn.speaker_id && turn.speaker_id !== 'UNKNOWN' ? turn.speaker_id : '';
        const normalized = { ...turn, speaker_id: speaker };
        const previous = merged[merged.length - 1];
        if (
          previous &&
          previous.speaker_id &&
          normalized.speaker_id &&
          previous.speaker_id === normalized.speaker_id &&
          normalized.start - previous.end <= 1.2
        ) {
          previous.end = Math.max(previous.end, normalized.end);
          previous.text = joinText(previous.text, normalized.text);
          continue;
        }
        merged.push(normalized);
      }
      return merged;
    }

    function joinText(left, right) {
      if (!left) return right || '';
      if (!right) return left || '';
      return `${left}${/[\u3400-\u9fff]$/.test(left) ? '' : ' '}${right}`;
    }

    function renderEvents(events) {
      if (!events.length) return '<div class="empty">没有事件</div>';
      return events.map((event) => `
        <div class="turn">
          <div class="time">#${event.chunk_id} ${event.start.toFixed(2)} - ${event.end.toFixed(2)}</div>
          <div class="speaker">${escapeHtml(event.event_type)}</div>
          <div class="text">${escapeHtml(event.speaker_id || '')} ${escapeHtml(event.text || event.message || '')}</div>
        </div>
      `).join('');
    }

    function setEmpty(id, text) {
      document.getElementById(id).innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
    }

    async function drawWaveform(file) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#eef2f6';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      try {
        const audioCtx = new AudioContext();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const samples = audioBuffer.getChannelData(0);
        const step = Math.max(1, Math.floor(samples.length / canvas.width));
        ctx.strokeStyle = '#0f766e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < canvas.width; x++) {
          let min = 1, max = -1;
          for (let i = 0; i < step; i++) {
            const v = samples[x * step + i] || 0;
            if (v < min) min = v;
            if (v > max) max = v;
          }
          const y1 = (1 + min) * canvas.height / 2;
          const y2 = (1 + max) * canvas.height / 2;
          ctx.moveTo(x, y1);
          ctx.lineTo(x, y2);
        }
        ctx.stroke();
        audioCtx.close();
      } catch (_) {
        ctx.fillStyle = '#667085';
        ctx.font = '14px sans-serif';
        ctx.fillText('当前浏览器无法预览该音频波形', 18, 44);
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[ch]));
    }
  </script>
</body>
</html>
"""


class WebAsrHandler(BaseHTTPRequestHandler):
    server_version = "m_asr_web/0.1"

    def do_GET(self) -> None:
        if self.path.startswith("/ws/live"):
            self._handle_live_websocket()
            return
        if self.path in {"/", "/index.html", "/live"}:
            self._send_html(LIVE_HTML_PAGE)
            return
        if self.path == "/upload":
            self._send_html(HTML_PAGE)
            return
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html", "/live"}:
            encoded = LIVE_HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            return
        if self.path == "/upload":
            encoded = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            return
        if self.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if self.path != "/api/transcribe":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return

        try:
            path = self._save_upload()
            try:
                result = transcribe_file(path, self.server.config_path)  # type: ignore[attr-defined]
            finally:
                with contextlib.suppress(OSError):
                    path.unlink()
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _handle_live_websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(HTTPStatus.BAD_REQUEST, "websocket upgrade required")
            return

        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        session: LiveWebSocketSession | None = None
        try:
            session = LiveWebSocketSession(self, self.server.config_path)  # type: ignore[attr-defined]
            session.run()
        except WebSocketClosed:
            return
        except Exception as exc:
            with contextlib.suppress(Exception):
                _ws_send_json(self, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if session is not None:
                session.close()

    def _save_upload(self) -> Path:
        content_type = self.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("request must be multipart/form-data")
        boundary = _get_boundary(content_type)
        content_length = int(self.headers.get("content-length", "0"))
        if content_length <= 0:
            raise ValueError("empty upload")
        if content_length > 512 * 1024 * 1024:
            raise ValueError("upload is larger than 512 MiB")

        body = self.rfile.read(content_length)
        upload = _extract_multipart_file(body, boundary, "audio")
        if upload is None:
            raise ValueError("missing audio file field")

        filename, data = upload
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        fd, raw_path = tempfile.mkstemp(prefix="m_asr_upload_", suffix=suffix)
        path = Path(raw_path)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def transcribe_file(path: Path, config_path: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    waveform, sample_rate = read_wav(path)
    pipeline = StreamingCascadePipeline(config)
    events: list[PipelineEvent] = list(pipeline.process_waveform(waveform, sample_rate))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    warning = ""
    if config.runtime.device == "cpu" or config.runtime.asr_provider == "cpu":
        warning = "CUDA 不可用或当前 sherpa-onnx 不支持 CUDA provider，本次已使用 CPU。"
    if not any(event.event_type == "chunk_finalized" for event in events):
        warning = "没有切出语音 chunk；请检查音频音量或 chunker.energy_reference。"
    elif not pipeline.transcript:
        warning = "已切出语音 chunk，但 ASR 没有返回最终文本。"

    return {
        "input": {"name": path.name, "sample_rate": sample_rate, "samples": int(len(waveform))},
        "runtime": {"device": config.runtime.device, "asr_provider": config.runtime.asr_provider},
        "backends": {"asr": pipeline.backends.asr, "speaker": pipeline.backends.speaker},
        "elapsed_ms": elapsed_ms,
        "warning": warning,
        "events": [_event_to_dict(event) for event in events],
        "transcript": [_turn_to_dict(turn) for turn in pipeline.transcript],
    }


class WebSocketClosed(Exception):
    pass


class LiveWebSocketSession:
    def __init__(self, handler: WebAsrHandler, config_path: str):
        self.handler = handler
        self.config = load_config(config_path)
        self.pipeline = StreamingCascadePipeline(self.config)
        self.asr_stream = self.pipeline.asr.create_streaming_session()
        self.executor = ThreadPoolExecutor(max_workers=self.config.runtime.max_workers)
        self.sample_rate = self.config.runtime.sample_rate
        self.live_chunk_id = 0
        self.current_start = 0.0
        self.current_text = ""
        self._speaker_futures: list[tuple[Any, str, Future[Any]]] = []

    def run(self) -> None:
        warning = ""
        if self.config.runtime.device == "cpu" or self.config.runtime.asr_provider == "cpu":
            warning = "CUDA 不可用或当前 sherpa-onnx 不支持 CUDA provider，本次实时识别使用 CPU。"
        _ws_send_json(
            self.handler,
            {
                "type": "ready",
                "runtime": {
                    "device": self.config.runtime.device,
                    "asr_provider": self.config.runtime.asr_provider,
                },
                "backends": {
                    "asr": self.pipeline.backends.asr,
                    "speaker": self.pipeline.backends.speaker,
                },
                "sample_rate": self.sample_rate,
                "warning": warning,
            },
        )

        while True:
            opcode, payload = _ws_read_frame(self.handler)
            if opcode == 0x8:
                raise WebSocketClosed()
            if opcode == 0x9:
                _ws_send_frame(self.handler, payload, opcode=0xA)
                continue
            if opcode == 0x1:
                self._handle_text(payload.decode("utf-8", errors="replace"))
                self._drain_speaker_futures(wait=False)
                continue
            if opcode == 0x2:
                self._handle_audio(payload)
                self._drain_speaker_futures(wait=False)

    def close(self) -> None:
        self._drain_speaker_futures(wait=True)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _handle_text(self, text: str) -> None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            _ws_send_json(self.handler, {"type": "error", "error": "invalid JSON control message"})
            return

        msg_type = message.get("type")
        if msg_type == "start":
            client_rate = int(message.get("sample_rate") or self.sample_rate)
            if client_rate != self.sample_rate:
                _ws_send_json(
                    self.handler,
                    {
                        "type": "error",
                        "error": f"client sample_rate must be {self.sample_rate}, got {client_rate}",
                    },
                )
            return
        if msg_type == "stop":
            self._flush()
            self._drain_speaker_futures(wait=True)
            _ws_send_json(self.handler, {"type": "stopped"})
            return
        _ws_send_json(self.handler, {"type": "error", "error": f"unknown control message: {msg_type}"})

    def _handle_audio(self, payload: bytes) -> None:
        if not payload:
            return
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        write = self.pipeline.audio_buffer.append(samples, self.sample_rate)
        partial = self.asr_stream.accept_waveform(write.waveform, self.sample_rate)
        if partial and partial != self.current_text:
            self.current_text = partial
            _ws_send_json(
                self.handler,
                {
                    "type": "event",
                    "event": _event_to_dict(
                        PipelineEvent(
                            "partial",
                            self.live_chunk_id,
                            self.current_start,
                            write.end_time,
                            speaker_id="",
                            text=partial,
                            confidence=0.0,
                        )
                    ),
                },
            )
        chunks = self.pipeline.chunker.accept(write.waveform)
        self._process_chunks(chunks)

    def _flush(self) -> None:
        chunks = self.pipeline.chunker.flush()
        self._process_chunks(chunks)
        if self.current_text:
            self._finalize_current_asr(self.pipeline.audio_buffer.total_samples / self.sample_rate)

    def _process_chunks(self, chunks: list[Any]) -> None:
        for chunk in chunks:
            _ws_send_json(
                self.handler,
                {
                    "type": "event",
                    "event": _event_to_dict(
                        PipelineEvent("chunk_finalized", chunk.chunk_id, chunk.start, chunk.end)
                    ),
                },
            )
            self._finalize_streaming_chunk(chunk)

    def _finalize_streaming_chunk(self, chunk: Any) -> None:
        final_text = self.asr_stream.finish() or self.current_text
        if final_text:
            _ws_send_json(
                self.handler,
                {
                    "type": "event",
                    "event": _event_to_dict(
                        PipelineEvent(
                            "final",
                            chunk.chunk_id,
                            chunk.start,
                            chunk.end,
                            speaker_id="",
                            text=final_text,
                            confidence=0.0,
                        )
                    ),
                },
            )
            future = self.executor.submit(self.pipeline._identify_speaker, chunk)
            self._speaker_futures.append((chunk, final_text, future))
        self.asr_stream.reset()
        self.live_chunk_id = chunk.chunk_id + 1
        self.current_start = chunk.end
        self.current_text = ""

    def _finalize_current_asr(self, end_time: float) -> None:
        final_text = self.asr_stream.finish() or self.current_text
        if final_text:
            _ws_send_json(
                self.handler,
                {
                    "type": "event",
                    "event": _event_to_dict(
                        PipelineEvent(
                            "final",
                            self.live_chunk_id,
                            self.current_start,
                            end_time,
                            speaker_id="",
                            text=final_text,
                            confidence=0.0,
                        )
                    ),
                },
            )
        self.asr_stream.reset()
        self.current_text = ""

    def _drain_speaker_futures(self, wait: bool) -> None:
        remaining: list[tuple[Any, str, Future[Any]]] = []
        for chunk, text, future in self._speaker_futures:
            if not wait and not future.done():
                remaining.append((chunk, text, future))
                continue
            try:
                speaker_result = future.result()
            except Exception as exc:
                _ws_send_json(
                    self.handler,
                    {
                        "type": "event",
                        "event": _event_to_dict(
                            PipelineEvent(
                                "error",
                                chunk.chunk_id,
                                chunk.start,
                                chunk.end,
                                message=f"speaker embedding failed: {type(exc).__name__}: {exc}",
                            )
                        ),
                    },
                )
                continue

            self.pipeline.transcript.append(
                TranscriptTurn(
                    start=chunk.start,
                    end=chunk.end,
                    speaker_id=speaker_result.speaker_id,
                    text=text,
                    confidence=speaker_result.confidence,
                )
            )
            _ws_send_json(
                self.handler,
                {
                    "type": "event",
                    "event": _event_to_dict(
                        PipelineEvent(
                            "speaker",
                            chunk.chunk_id,
                            chunk.start,
                            chunk.end,
                            speaker_id=speaker_result.speaker_id,
                            confidence=speaker_result.confidence,
                        )
                    ),
                },
            )
        self._speaker_futures = remaining


def _ws_read_exact(handler: WebAsrHandler, size: int) -> bytes:
    data = handler.rfile.read(size)
    if len(data) != size:
        raise WebSocketClosed()
    return data


def _ws_read_frame(handler: WebAsrHandler) -> tuple[int, bytes]:
    first, second = _ws_read_exact(handler, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _ws_read_exact(handler, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _ws_read_exact(handler, 8))[0]
    mask = _ws_read_exact(handler, 4) if masked else b""
    payload = _ws_read_exact(handler, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def _ws_send_json(handler: WebAsrHandler, data: dict[str, Any]) -> None:
    _ws_send_frame(handler, json.dumps(data, ensure_ascii=False).encode("utf-8"), opcode=0x1)


def _ws_send_frame(handler: WebAsrHandler, payload: bytes, opcode: int = 0x1) -> None:
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    handler.wfile.write(header + payload)
    handler.wfile.flush()


def _event_to_dict(event: PipelineEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "chunk_id": event.chunk_id,
        "start": event.start,
        "end": event.end,
        "speaker_id": event.speaker_id,
        "text": event.text,
        "confidence": event.confidence,
        "message": event.message,
    }


def _turn_to_dict(turn: TranscriptTurn) -> dict[str, Any]:
    return {
        "start": turn.start,
        "end": turn.end,
        "speaker_id": turn.speaker_id,
        "text": turn.text,
        "confidence": turn.confidence,
    }


def _get_boundary(content_type: str) -> bytes:
    for part in content_type.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key.lower() == "boundary":
            return value.strip('"').encode("utf-8")
    raise ValueError("multipart boundary not found")


def _extract_multipart_file(
    body: bytes,
    boundary: bytes,
    field_name: str,
) -> tuple[str, bytes] | None:
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip()
        header_blob, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next(
            (header for header in headers if header.lower().startswith("content-disposition:")),
            "",
        )
        attrs = _parse_disposition_attrs(disposition)
        if attrs.get("name") != field_name:
            continue
        return attrs.get("filename", "audio.wav"), data.rstrip(b"\r\n")
    return None


def _parse_disposition_attrs(header: str) -> dict[str, str]:
    _, _, value = header.partition(":")
    attrs: dict[str, str] = {}
    for part in value.split(";"):
        key, sep, raw = part.strip().partition("=")
        if sep:
            attrs[key.lower()] = raw.strip().strip('"')
    return attrs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run m_asr web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), WebAsrHandler)
    server.config_path = args.config  # type: ignore[attr-defined]
    print(f"[web] serving http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
