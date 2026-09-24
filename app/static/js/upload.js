const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const idleEl = dropzone.querySelector(".dropzone-idle");
const busyEl = dropzone.querySelector(".dropzone-busy");
const busyTitle = document.getElementById("busy-title");
const errorMsg = document.getElementById("error-msg");

function setBusy(busy) {
  idleEl.hidden = busy;
  busyEl.hidden = !busy;
  dropzone.classList.toggle("busy", busy);
  fileInput.disabled = busy;
}

async function upload(files) {
  if (!files.length) return;
  errorMsg.hidden = true;
  busyTitle.textContent = files.length === 1 ? "Finding bands…" : `Finding bands in ${files.length} images…`;
  setBusy(true);

  const formData = new FormData();
  for (const f of files) formData.append("images", f);

  try {
    const resp = await fetch("/api/projects", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Upload failed.");
    window.location.href = `/p/${data.project_id}`;
  } catch (err) {
    setBusy(false);
    fileInput.value = "";
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  }
}

fileInput.addEventListener("change", () => upload(fileInput.files));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  if (!dropzone.classList.contains("busy")) upload(e.dataTransfer.files);
});
