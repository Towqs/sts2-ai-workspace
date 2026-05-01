# -*- coding: utf-8 -*-
"""Patch control_panel.py with all needed fixes - using line-based approach."""

path = r"d:\2024 fa fan\XJ12615\STS2_AI_Workspace\AI_Training\control_panel.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

content = "".join(lines)

# === 1. Fix jsString ===
old = 'function jsString(value) {\n  return JSON.stringify(String(value ?? ""));\n}'
new = "function jsString(value) {\n  return \"'\" + String(value ?? \"\").replace(/\\\\\\\\/g, \"\\\\\\\\\\\\\\\\\").replace(/'/g, \"\\\\\\\\'\") + \"'\";\n}"
assert old in content, f"jsString not found"
content = content.replace(old, new, 1)

# === 2. Add backend functions ===
marker = '    return {"status": "ok", "package": model_package_status(model_id)}\n\n\ndef model_registry_status'
assert marker in content, "backend insert not found"
backend_code = '''    return {"status": "ok", "package": model_package_status(model_id)}


def delete_model_package(model_id):
    model_id = safe_model_id(model_id)
    if not model_id:
        return {"status": "error", "error": "invalid model_id"}
    active = safe_model_id(read_control().get("active_model_id"))
    if model_id == active:
        return {"status": "error", "error": "\u65e0\u6cd5\u5220\u9664\u5f53\u524d\u542f\u7528\u7684\u6a21\u578b"}
    root = MODEL_ZOO_DIR / model_id
    if not root.exists() or not root.is_dir():
        return {"status": "error", "error": "\u672a\u627e\u5230\u6a21\u578b\u5305"}
    shutil.rmtree(root, ignore_errors=True)
    return {"status": "ok"}


def import_model_package(zip_data, label="", activate=False):
    import io
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data))
    except Exception as exc:
        return {"status": "error", "error": f"\u65e0\u6cd5\u8bfb\u53d6 zip \u6587\u4ef6\uff1a{exc}"}
    names = zf.namelist()
    top_dirs = set()
    for n in names:
        parts = n.replace("\\\\", "/").split("/")
        if len(parts) > 1 and parts[0]:
            top_dirs.add(parts[0])
    strip_prefix = ""
    if len(top_dirs) == 1:
        strip_prefix = list(top_dirs)[0] + "/"
    found_artifacts = set()
    for dirname, filenames in MODEL_ARTIFACTS.items():
        for filename in filenames:
            check_path = f"{strip_prefix}{dirname}/{filename}"
            if check_path in names:
                found_artifacts.add(f"{dirname}/{filename}")
    if not found_artifacts:
        return {"status": "error", "error": "zip \u4e2d\u672a\u627e\u5230\u6709\u6548\u7684\u6a21\u578b\u6587\u4ef6\uff08ProcessedParams/ProcessedMacroParams\uff09"}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_id = f"imported_{stamp}_{uuid.uuid4().hex[:6]}"
    root = MODEL_ZOO_DIR / package_id
    MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel_path = info.filename.replace("\\\\", "/")
            if strip_prefix and rel_path.startswith(strip_prefix):
                rel_path = rel_path[len(strip_prefix):]
            if not rel_path:
                continue
            dest = root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        return {"status": "error", "error": f"\u89e3\u538b\u5931\u8d25\uff1a{exc}"}
    finally:
        zf.close()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path, {})
    created_at = datetime.now().isoformat(timespec="seconds")
    manifest.update({
        "id": package_id,
        "label": str(label or manifest.get("label") or f"\u5bfc\u5165\u6a21\u578b {created_at.replace('T', ' ')}").strip()[:80],
        "description": str(manifest.get("description") or "\u4ece zip \u5bfc\u5165\u7684\u6a21\u578b\u5305\u3002").strip()[:300],
        "created_at": manifest.get("created_at") or created_at,
        "source": "imported",
        "retention": "manual",
        "pinned": True,
    })
    write_json(manifest_path, manifest)
    if activate:
        set_active_model_id(package_id)
    return {
        "status": "ok",
        "model_id": package_id,
        "label": manifest["label"],
        "package": model_package_status(package_id),
    }


def model_registry_status'''
content = content.replace(marker, backend_code, 1)

# === 3. Add render guard ===
old_render = 'function renderModelHealth(models, aiProcess, control, runtime, monsterProfiles) {\n  const combat = models.combat || {};'
new_render = '''function renderModelHealth(models, aiProcess, control, runtime, monsterProfiles) {
  const modelHealthDiv = document.getElementById("modelHealth");
  if (modelHealthDiv && (modelHealthDiv.contains(document.activeElement) || modelHealthDiv.querySelector('input[data-editing="1"]'))) {
    return;
  }
  let detailsOpen = false;
  const oldDetails = document.getElementById("modelPackageDetails");
  if (oldDetails && oldDetails.open) {
    detailsOpen = true;
  }
  const combat = models.combat || {};'''
assert old_render in content, "renderModelHealth not found"
content = content.replace(old_render, new_render, 1)

# === 4. Fix row template - find via unique line and replace surrounding ===
# Find the button lines and replace them
old_buttons = '        <button onclick="renameModelPackage(${jsString(pkg.id)})">\u4fdd\u5b58\u540d\u79f0</button>\n        <button onclick="pinModelPackage(${jsString(pkg.id)})" ${isPinned ? "disabled" : ""}>\u6c38\u4e45\u4fdd\u7559</button>\n      </td>'
new_buttons = '        <div style="display:flex;gap:4px;flex-wrap:wrap">\n          <button class="small" onclick="renameModelPackage(${jsString(pkg.id)})">\u4fdd\u5b58\u540d\u79f0</button>\n          <button class="small" onclick="pinModelPackage(${jsString(pkg.id)})" ${isPinned ? "disabled" : ""}>\u6c38\u4e45\u4fdd\u7559</button>\n          <button class="small off" onclick="deleteModelPackage(${jsString(pkg.id)})" ${pkg.id === activeModelId ? "disabled" : ""}>\u5220\u9664</button>\n        </div>\n      </td>'
assert old_buttons in content, f"old buttons not found"
content = content.replace(old_buttons, new_buttons, 1)

# Fix the td with code to add title and truncation  
old_code_td = '      <td><code>${escapeHtml(pkg.id)}</code><br><span class="fine">${escapeHtml(pkg.created_at || "")}</span></td>'
new_code_td = '      <td title="${escapeHtml(pkg.id)}"><code style="max-width:120px;overflow:hidden;text-overflow:ellipsis;display:inline-block;white-space:nowrap">${escapeHtml(pkg.id)}</code><br><span class="fine">${escapeHtml((pkg.created_at || "").split("T")[0])}</span></td>'
assert old_code_td in content, "old code td not found"
content = content.replace(old_code_td, new_code_td, 1)

# Fix input to add onfocus/onblur
old_input = '        <input id="${escapeHtml(rowId)}" type="text" value="${escapeHtml(pkg.label || pkg.id)}" style="min-width:180px;width:100%;max-width:260px">\n        <div class="fine">${escapeHtml(pkg.description || "")}</div>'
new_input = '        <input id="${escapeHtml(rowId)}" type="text" value="${escapeHtml(pkg.label || pkg.id)}"\n               style="min-width:140px;width:100%;max-width:220px"\n               onfocus="this.dataset.editing=\'1\'" onblur="this.dataset.editing=\'\'">\n        <div class="fine" style="font-size:11px;opacity:0.7">${escapeHtml(pkg.description || "")}</div>'
assert old_input in content, "old input not found"
content = content.replace(old_input, new_input, 1)

# Compact the sample/source columns
old_sample = '      <td>\u6218\u6597 ${summary.combat_samples || 0} / \u5019\u9009 ${summary.candidate_rows || 0}<br><span class="fine">\u5b8f\u89c2 ${summary.macro_samples || 0}\uff0cAI ${summary.include_ai ? "\u5df2\u7eb3\u5165" : "\u672a\u7eb3\u5165"}</span></td>\n      <td><span class="fine">C: H${combatSources.human || 0}/AI${combatSources.ai || 0}<br>M: H${macroSources.human || 0}/AI${macroSources.ai || 0}</span></td>'
new_sample = '      <td style="white-space:nowrap">\u6218:${summary.combat_samples || 0} / \u5019:${summary.candidate_rows || 0}<br><span class="fine">\u5b8f:${summary.macro_samples || 0}${summary.include_ai ? "/AI" : ""}</span></td>\n      <td class="fine">C:H${combatSources.human || 0}/A${combatSources.ai || 0}<br>M:H${macroSources.human || 0}/A${macroSources.ai || 0}</td>'
assert old_sample in content, "old sample cols not found"
content = content.replace(old_sample, new_sample, 1)

# Compact status pills
old_status = '        <span class="pill ${pkg.complete ? "on" : "off"}">${pkg.complete ? "\u5b8c\u6574" : "\u7f3a\u6587\u4ef6"}</span><br>\n        <span class="pill ${isPinned ? "on" : "info"}">${isPinned ? "\u6c38\u4e45\u4fdd\u7559" : "\u81ea\u52a8\u4fdd\u7559"}</span><br>\n        <span class="fine">${formatBytes(pkg.size || 0)}</span>'
new_status = '        <span class="pill ${pkg.complete ? "on" : "off"}" style="padding:1px 4px;font-size:10px">${pkg.complete ? "\u5b8c\u6574" : "\u7f3a\u9879"}</span>\n        <span class="pill ${isPinned ? "on" : "info"}" style="padding:1px 4px;font-size:10px">${isPinned ? "\u6c38\u4e45\u4fdd\u7559" : "\u81ea\u52a8\u6e05\u7406"}</span><br>\n        <span class="fine" style="font-size:10px">${formatBytes(pkg.size || 0)}</span>'
assert old_status in content, "old status pills not found"
content = content.replace(old_status, new_status, 1)

# === 5. Replace table-wrap with details + import UI ===
old_table_area = '      <div id="modelSwitchResult" class="fine" style="margin-top:6px">\u5f53\u524d\u542f\u7528\uff1a${escapeHtml(registry.active_label || activeModelId)}\u3002\u5207\u6362\u540e\u9700\u8981\u91cd\u542f AI \u8fdb\u7a0b\u3002</div>\n      <div class="fine" style="margin-top:6px">\u8bad\u7ec3\u5b8c\u6210\u4f1a\u81ea\u52a8\u4fdd\u5b58\u6a21\u578b\u5feb\u7167\uff0c\u53ea\u4fdd\u7559\u6700\u8fd1 ${autoKeepLimit} \u4e2a\uff1b\u70b9\u201c\u6c38\u4e45\u4fdd\u7559\u201d\u7684\u6a21\u578b\u4e0d\u4f1a\u88ab\u81ea\u52a8\u6e05\u7406\u3002</div>\n    </div>\n    <div class="table-wrap" style="margin-top:10px">\n      <table>\n        <thead><tr><th>ID</th><th>\u540d\u79f0</th><th>\u6837\u672c</th><th>\u6765\u6e90</th><th>\u72b6\u6001</th><th>\u64cd\u4f5c</th></tr></thead>\n        <tbody>${packageRows || \'<tr><td colspan=6>\u6682\u65e0\u53ef\u5207\u6362\u6a21\u578b\u5305</td></tr>\'}</tbody>\n      </table>\n    </div>`;'

new_table_area = '      <div class="field" style="margin-top:10px">\n        <span>\u5bfc\u5165\u6a21\u578b\u5305\uff08zip\uff09</span>\n        <input id="importModelFile" type="file" accept=".zip" style="font-size:12px">\n      </div>\n      <div class="button-row" style="margin-top:8px">\n        <button onclick="importModelPackage()">\u5bfc\u5165\u6a21\u578b</button>\n      </div>\n      <div id="modelSwitchResult" class="fine" style="margin-top:6px">\u5f53\u524d\u542f\u7528\uff1a${escapeHtml(registry.active_label || activeModelId)}\u3002\u5207\u6362\u540e\u9700\u8981\u91cd\u542f AI \u8fdb\u7a0b\u3002</div>\n      <div class="fine" style="margin-top:6px">\u8bad\u7ec3\u5b8c\u6210\u4f1a\u81ea\u52a8\u4fdd\u5b58\u6a21\u578b\u5feb\u7167\uff0c\u53ea\u4fdd\u7559\u6700\u8fd1 ${autoKeepLimit} \u4e2a\uff1b\u70b9\u201c\u6c38\u4e45\u4fdd\u7559\u201d\u7684\u6a21\u578b\u4e0d\u4f1a\u88ab\u81ea\u52a8\u6e05\u7406\u3002</div>\n    </div>\n    <details id="modelPackageDetails" class="more-panel" ${detailsOpen ? "open" : ""}>\n      <summary>\u6a21\u578b\u5305\u5217\u8868\u7ba1\u7406\uff08${packages.length} \u4e2a\uff09</summary>\n      <div class="table-wrap" style="max-height:none;overflow-x:auto;overflow-y:visible">\n        <table>\n          <thead><tr><th>ID</th><th>\u540d\u79f0</th><th>\u6837\u672c</th><th>\u6765\u6e90</th><th>\u72b6\u6001</th><th>\u64cd\u4f5c</th></tr></thead>\n          <tbody>${packageRows || \'<tr><td colspan=6>\u6682\u65e0\u53ef\u5207\u6362\u6a21\u578b\u5305</td></tr>\'}</tbody>\n        </table>\n      </div>\n    </details>`;'
assert old_table_area in content, "old table area not found"
content = content.replace(old_table_area, new_table_area, 1)

# === 6. Add deleteModelPackage/importModelPackage JS + fix renameModelPackage ===
old_rename_fn = 'async function renameModelPackage(model_id){\n  const resultEl = document.getElementById("modelSwitchResult");\n  const input = document.getElementById(`modelLabel_${model_id}`);\n  const label = input ? input.value.trim() : "";\n  if (!model_id || !label) {\n    if (resultEl) resultEl.textContent = "\u6a21\u578b\u540d\u79f0\u4e0d\u80fd\u4e3a\u7a7a\u3002";\n    return;\n  }\n  const result = await api("/api/model/update", {model_id, label});\n  if (resultEl) resultEl.textContent = result.status === "ok" ? "\u6a21\u578b\u540d\u79f0\u5df2\u4fdd\u5b58\u3002" : (result.error || "\u4fdd\u5b58\u5931\u8d25");\n  refresh();\n}\nasync function pinModelPackage'

new_rename_fn = 'async function renameModelPackage(model_id){\n  const resultEl = document.getElementById("modelSwitchResult");\n  const input = document.getElementById(`modelLabel_${model_id}`);\n  const label = input ? input.value.trim() : "";\n  if (input) input.dataset.editing = "";\n  if (!model_id || !label) {\n    if (resultEl) resultEl.textContent = "\u6a21\u578b\u540d\u79f0\u4e0d\u80fd\u4e3a\u7a7a\u3002";\n    return;\n  }\n  const result = await api("/api/model/update", {model_id, label});\n  if (resultEl) resultEl.textContent = result.status === "ok" ? "\u6a21\u578b\u540d\u79f0\u5df2\u4fdd\u5b58\u3002" : (result.error || "\u4fdd\u5b58\u5931\u8d25");\n  refresh();\n}\nasync function deleteModelPackage(model_id){\n  if (!confirm(`\u786e\u5b9a\u8981\u5f7b\u5e95\u5220\u9664\u6a21\u578b\u5305 ${model_id} \u5417\uff1f\u6b64\u64cd\u4f5c\u4e0d\u53ef\u6062\u590d\u3002`)) return;\n  const resultEl = document.getElementById("modelSwitchResult");\n  if (resultEl) resultEl.textContent = "\u6b63\u5728\u5220\u9664...";\n  const result = await api("/api/model/delete", {model_id});\n  if (resultEl) resultEl.textContent = result.status === "ok" ? "\u6a21\u578b\u5305\u5df2\u5220\u9664\u3002" : (result.error || "\u5220\u9664\u5931\u8d25");\n  refresh();\n}\nasync function importModelPackage(){\n  const resultEl = document.getElementById("modelSwitchResult");\n  const fileInput = document.getElementById("importModelFile");\n  if (!fileInput || !fileInput.files || !fileInput.files.length) {\n    if (resultEl) resultEl.textContent = "\u8bf7\u5148\u9009\u62e9\u4e00\u4e2a zip \u6587\u4ef6\u3002";\n    return;\n  }\n  const file = fileInput.files[0];\n  if (resultEl) resultEl.textContent = `\u6b63\u5728\u5bfc\u5165 ${file.name}\u2026`;\n  try {\n    const buf = await file.arrayBuffer();\n    const resp = await fetch("/api/model/import", {\n      method: "POST",\n      headers: {"Content-Type": "application/octet-stream"},\n      body: buf,\n    });\n    const result = await resp.json();\n    if (result.status === "ok") {\n      fileInput.value = "";\n      if (resultEl) resultEl.textContent = `\u5bfc\u5165\u6210\u529f\uff1a${result.label || result.model_id}`;\n    } else {\n      if (resultEl) resultEl.textContent = result.error || "\u5bfc\u5165\u5931\u8d25";\n    }\n  } catch(e) {\n    if (resultEl) resultEl.textContent = "\u5bfc\u5165\u5931\u8d25\uff1a" + e.message;\n  }\n  refresh();\n}\nasync function pinModelPackage'
assert old_rename_fn in content, "old rename fn not found"
content = content.replace(old_rename_fn, new_rename_fn, 1)

# === 7. Add API endpoints ===
old_export_ep = '            elif self.path == "/api/export":\n                self._json(200, export_database_package())'
new_export_ep = '            elif self.path == "/api/model/delete":\n                self._json(200, delete_model_package(body.get("model_id")))\n            elif self.path == "/api/export":\n                self._json(200, export_database_package())'
assert old_export_ep in content, "/api/export not found"
content = content.replace(old_export_ep, new_export_ep, 1)

old_post_start = '    def do_POST(self):\n        try:\n            body = self._body()'
new_post_start = '    def do_POST(self):\n        try:\n            if self.path == "/api/model/import":\n                length = int(self.headers.get("Content-Length", "0"))\n                if length < 1:\n                    self._json(400, {"status": "error", "error": "\u6ca1\u6709\u4e0a\u4f20\u6587\u4ef6"})\n                    return\n                zip_data = self.rfile.read(length)\n                label = self.headers.get("X-Model-Label", "")\n                activate = self.headers.get("X-Model-Activate", "") == "1"\n                self._json(200, import_model_package(zip_data, label=label, activate=activate))\n                return\n            body = self._body()'
assert old_post_start in content, "do_POST not found"
content = content.replace(old_post_start, new_post_start, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("All patches applied successfully!")
