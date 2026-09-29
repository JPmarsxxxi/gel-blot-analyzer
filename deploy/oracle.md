# Deploy on Oracle Cloud (free)

This puts the app on a free Oracle Cloud server with HTTPS and projects that
survive restarts. It takes about 30 minutes, most of it waiting.

## 1. Create the server (once)

1. Sign up at https://www.oracle.com/cloud/free/. The card is only for
   identity checks. Pick your home region carefully: it can't be changed.
2. In the console, open **Compute > Instances > Create instance**.
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** Ampere, `VM.Standard.A1.Flex`, 2 OCPUs and 12 GB memory
     (up to 4 and 24 GB stays free).
   - **Networking:** keep the defaults, with a public IPv4 address.
   - **SSH keys:** "Generate a key pair for me", then **Save private key**.
   - Click **Create**. If it says "Out of capacity", try another
     availability domain or fewer OCPUs, or try again later.
3. When it's running, copy its **Public IP address** from the instance page.

## 2. Open the web ports (once)

On the instance page, click the **subnet**, then its **Default Security List**,
then **Add Ingress Rules**:

- Source CIDR: `0.0.0.0/0`
- IP protocol: TCP
- Destination port range: `80,443`

## 3. Install the app

On your PC, open PowerShell and connect (use your key's path and the IP):

```powershell
ssh -i C:\Users\User\Downloads\ssh-key.key ubuntu@YOUR_IP
```

If it says the key's permissions are too open, run this once and retry:

```powershell
icacls C:\Users\User\Downloads\ssh-key.key /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

Then, on the server:

```bash
curl -fsSL https://raw.githubusercontent.com/JPmarsxxxi/gel-blot-analyzer/main/deploy/oracle-setup.sh | bash
```

The first run takes about 10 minutes. It ends by printing the address, which
looks like `https://203.0.113.7.sslip.io`. Open it; the first load can take a
minute while the HTTPS certificate is issued.

## Updating

After merging changes on GitHub, connect again and run the same `curl ... | bash`
line. Projects are kept.

## Good to know

- Projects live in `/srv/gel-data` on the server. Copy that folder to back
  them up.
- The address contains the server's IP. If you ever delete and recreate the
  server, the IP and therefore the address change. For a permanent name, point
  a domain at the IP and run `DOMAIN=your.domain bash ~/gel-blot-analyzer/deploy/oracle-setup.sh`.
- Oracle may reclaim Always Free servers that stay almost idle for a week.
  Check their current policy if the app gets little traffic.
