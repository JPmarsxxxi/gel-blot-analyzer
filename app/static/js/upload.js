const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const uploadBtn = document.getElementById("upload-btn");
const form = document.getElementById("upload-form");
const errorMsg = document.getElementById("error-msg");
const statusMsg = document.getElementById("status-msg");
const dropzone = document.getElementById("dropzone");

function renderFileList() {
  fileList.innerHTML = "";
  const files = fileInput.files;
  for (const f of files) {
    const li = document.createElement("li");
    li.textContent = f.name;
    fileList.appendChild(li);
  }
  uploadBtn.disabled = files.length === 0;
}

fileInput.addEventListener("change", renderFileList);

["dragover", "dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => e.preventDefault());
});
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    renderFileList();
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorMsg.hidden = true;
  statusMsg.hidden = false;
  uploadBtn.disabled = true;

  const formData = new FormData();
  for (const f of fileInput.files) {
    formData.append("images", f);
  }

  try {
    const resp = await fetch("/api/projects", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || "Upload failed.");
    }
    window.location.href = `/p/${data.project_id}`;
  } catch (err) {
    statusMsg.hidden = true;
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
    uploadBtn.disabled = false;
  }
});
