/* CatFace Search front end.
 *
 * Plain DOM, no framework and no build step: this page must work from a checkout with no toolchain.
 * The server renders everything a reader needs (model facts, the limitations section); this script
 * only adds the interactive search, so a JavaScript failure degrades to a readable static page
 * rather than a blank one.
 *
 * The margin is rendered as a bar and labelled. That is the whole point of the result view: the
 * project's own error analysis found every remaining failure sits on a near tie, so a result shown
 * without its margin would imply a confidence the system does not have.
 */
(function () {
  "use strict";

  var form = document.getElementById("search-form");
  if (!form) { return; }

  var dropzone = document.getElementById("dropzone");
  var input = document.getElementById("file-input");
  var preview = document.getElementById("preview");
  var submit = document.getElementById("submit");
  var clear = document.getElementById("clear");
  var progress = document.getElementById("progress");
  var errorBox = document.getElementById("error");
  var results = document.getElementById("results");
  var verdict = document.getElementById("verdict");
  var grid = document.getElementById("match-grid");
  var topK = document.getElementById("top-k");
  var aggregation = document.getElementById("identity-aggregation");

  var selected = null;
  var objectUrl = null;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function setSelected(file) {
    if (!file) { return; }
    if (!/^image\//.test(file.type)) {
      showError("这不是图片文件（浏览器报告的 MIME 类型为 " + (file.type || "未知") + "）。");
      return;
    }
    clearError();
    selected = file;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); }
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.hidden = false;
    dropzone.querySelector(".dropzone-inner").hidden = true;
  }

  function reset() {
    selected = null;
    input.value = "";
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    preview.hidden = true;
    preview.removeAttribute("src");
    dropzone.querySelector(".dropzone-inner").hidden = false;
    results.hidden = true;
    grid.innerHTML = "";
    verdict.innerHTML = "";
    clearError();
  }

  dropzone.addEventListener("click", function () { input.click(); });
  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
  });
  input.addEventListener("change", function () { setSelected(input.files[0]); });

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (event) {
    if (event.dataTransfer && event.dataTransfer.files.length) {
      setSelected(event.dataTransfer.files[0]);
    }
  });

  clear.addEventListener("click", reset);

  function renderVerdict(data) {
    verdict.innerHTML = "";
    var head = document.createElement("div");
    head.className = "verdict-head";

    var identity = document.createElement("span");
    identity.className = "verdict-identity";
    identity.textContent = data.predicted_identity || "无结果";

    var meta = document.createElement("span");
    meta.className = "verdict-meta";
    var sim = data.top_similarity === null ? "—" : data.top_similarity.toFixed(4);
    var margin = data.margin === null ? "—" : data.margin.toFixed(4);
    meta.textContent = "top-1 余弦 " + sim + " · margin " + margin +
      " · 嵌入 " + data.timing.embedding_ms.toFixed(0) + " ms · 检索 " +
      data.timing.search_ms.toFixed(0) + " ms · " + data.descriptor_dim + " 维";

    head.appendChild(identity);
    head.appendChild(meta);
    verdict.appendChild(head);

    var block = document.createElement("div");
    block.className = "margin-block";
    var label = document.createElement("div");
    label.className = "margin-label";
    var left = document.createElement("span");
    left.textContent = "margin = 正确身份相似度 − 最强错误身份相似度";
    var right = document.createElement("span");
    right.textContent = margin;
    label.appendChild(left);
    label.appendChild(right);

    var track = document.createElement("div");
    track.className = "margin-track";
    var fill = document.createElement("div");
    fill.className = "margin-fill";
    /* margin lives in [-1, 1] in principle; scale it so 0 lands at the middle of the track and the
       sign is visible. A negative margin means the top-1 identity is not actually ahead. */
    var value = data.margin === null ? 0 : data.margin;
    /* A null margin has a specific meaning: every returned match carried the same identity, so
       there is no different identity to compare against. Saying "—" alone would read as "unknown"
       when the answer is actually "there is nothing to compare". */
    if (data.margin === null) {
      var sameOnly = document.createElement("p");
      sameOnly.className = "lede";
      sameOnly.style.marginTop = "10px";
      sameOnly.textContent = "返回的前 " + data.matches.length +
        " 名全部是同一身份，没有可对比的其他身份，因此无法给出 margin。" +
        "这通常意味着上传的照片本身就在图库中。";
      block.appendChild(sameOnly);
      verdict.appendChild(block);
      return;
    }
    var width = Math.min(Math.abs(value), 1) * 50;
    fill.style.width = width + "%";
    fill.style.marginLeft = value < 0 ? (50 - width) + "%" : "50%";
    if (value < 0.05) { fill.classList.add(value < 0 ? "negative" : "low"); }
    track.appendChild(fill);

    block.appendChild(label);
    block.appendChild(track);

    if (value < 0.05) {
      var note = document.createElement("p");
      note.className = "lede";
      note.style.marginTop = "10px";
      note.textContent = value < 0
        ? "注意：margin 为负，说明排名第一的身份并不领先。本项目实测的错例全部属于这种情况。"
        : "注意：margin 很小，说明两个身份几乎并列。本项目实测的错例全部属于这种情况。";
      block.appendChild(note);
    }
    verdict.appendChild(block);
  }

  function renderMatches(matches) {
    grid.innerHTML = "";
    matches.forEach(function (match) {
      var item = document.createElement("li");
      item.className = "match";

      var img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "匹配到的猫脸：" + match.identity;
      img.src = "/api/gallery/" + encodeURIComponent(match.image_id);
      img.addEventListener("error", function () {
        img.replaceWith(Object.assign(document.createElement("div"), {
          className: "match-body",
          textContent: "该图库图片在当前部署中不可访问",
          style: "aspect-ratio:1;display:grid;place-items:center;color:var(--muted);font-size:13px;"
        }));
      });

      var body = document.createElement("div");
      body.className = "match-body";

      var rank = document.createElement("span");
      rank.className = "match-rank";
      rank.textContent = "#" + match.rank;

      var identity = document.createElement("span");
      identity.className = "match-identity";
      identity.textContent = match.identity;

      var sim = document.createElement("span");
      sim.className = "match-sim";
      sim.textContent = "相似度 " + match.similarity.toFixed(4);

      var bar = document.createElement("div");
      bar.className = "match-bar";
      var barFill = document.createElement("span");
      barFill.style.width = Math.max(0, Math.min(1, match.similarity)) * 100 + "%";
      bar.appendChild(barFill);

      body.appendChild(rank);
      body.appendChild(identity);
      body.appendChild(sim);
      body.appendChild(bar);
      if (match.identity_consensus >= 0.5) {
        var agree = document.createElement("span");
        agree.className = "match-agree";
        agree.textContent = "与前 " + matches.length + " 名中 " +
          Math.round(match.identity_consensus * 100) + "% 同一身份";
        body.appendChild(agree);
      }

      item.appendChild(img);
      item.appendChild(body);
      grid.appendChild(item);
    });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!selected) {
      showError("请先选择一张猫脸照片。");
      return;
    }
    clearError();
    progress.hidden = false;
    submit.disabled = true;

    var body = new FormData();
    body.append("file", selected, selected.name || "query.jpg");
    body.append("top_k", topK.value || "10");
    body.append("identity_aggregation", aggregation.checked ? "true" : "false");

    fetch("/api/search", { method: "POST", body: body })
      .then(function (response) {
        return response.json().catch(function () {
          throw new Error("服务返回了非 JSON 响应（HTTP " + response.status + "）");
        }).then(function (payload) {
          if (!response.ok) {
            var detail = payload && payload.detail ? payload.detail : "HTTP " + response.status;
            throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
          }
          return payload;
        });
      })
      .then(function (data) {
        renderVerdict(data);
        renderMatches(data.matches);
        results.hidden = false;
        results.scrollIntoView({ behavior: "smooth", block: "start" });
      })
      .catch(function (error) {
        showError("检索失败：" + error.message);
      })
      .finally(function () {
        progress.hidden = true;
        submit.disabled = false;
      });
  });
})();
