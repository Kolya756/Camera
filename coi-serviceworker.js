self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{
  if(e.request.mode!=='navigate')return;
  e.respondWith(fetch(e.request).then(r=>{
    if(r.status===0||r.type==='opaqueredirect')return r;
    const h=new Headers(r.headers);
    h.set('Cross-Origin-Embedder-Policy','require-corp');
    h.set('Cross-Origin-Opener-Policy','same-origin');
    return new Response(r.body,{status:r.status,statusText:r.statusText,headers:h});
  }));
});
