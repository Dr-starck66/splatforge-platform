function sha256(text){
  const bytes=Array.from(new TextEncoder().encode(text));
  const bitLen=bytes.length*8;bytes.push(128);while(bytes.length%64!==56)bytes.push(0);
  const hi=Math.floor(bitLen/4294967296),lo=bitLen>>>0;
  for(const w of [hi,lo])for(let j=3;j>=0;j--)bytes.push((w>>>(j*8))&255);
  const primes=[];for(let n=2;primes.length<64;n++){let yes=true;for(let d=2;d*d<=n;d++)if(n%d===0){yes=false;break;}if(yes)primes.push(n);}
  const frac=x=>(x-Math.floor(x))*4294967296>>>0;
  const h=primes.slice(0,8).map(p=>frac(Math.sqrt(p))),k=primes.map(p=>frac(Math.cbrt(p)));
  const rr=(v,n)=>(v>>>n)|(v<<(32-n));
  for(let pos=0;pos<bytes.length;pos+=64){
    const w=new Array(64);for(let i=0;i<16;i++){const j=pos+4*i;w[i]=((bytes[j]<<24)|(bytes[j+1]<<16)|(bytes[j+2]<<8)|bytes[j+3])>>>0;}
    for(let i=16;i<64;i++){const a=w[i-15],b=w[i-2];w[i]=(w[i-16]+(rr(a,7)^rr(a,18)^(a>>>3))+w[i-7]+(rr(b,17)^rr(b,19)^(b>>>10)))>>>0;}
    let [a,b,c,d,e,f,g,z]=h;
    for(let i=0;i<64;i++){
      const t1=(z+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[i]+w[i])>>>0;
      const t2=((rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c)))>>>0;
      z=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;
    }
    for(const [i,v] of [a,b,c,d,e,f,g,z].entries())h[i]=(h[i]+v)>>>0;
  }
  return h.map(x=>x.toString(16).padStart(8,'0')).join('');
}
const $=id=>document.getElementById(id);
const cases={nominal:'Nominal',dust:'Poussière',reactor:'Énergie',combined:'Crise combinée',unexpected:'Choc imprévu',water_leak:'Fuite d’eau',robot:'Robot'};
let report=null,lastEvidence=null,angle=.58,dragging=false,px=0;
const num=x=>Number(x).toLocaleString('fr-FR',{maximumFractionDigits:2,minimumFractionDigits:2});
function colorDelta(node,v){node.textContent=`${v>=0?'+':''}${num(v)} pt`;node.style.color=v<0?'#f2a8a1':'#88e6b4';}
function renderBars(rows){$('bars').replaceChildren(...rows.map(r=>{const el=document.createElement('div');el.className='bar';const label=document.createElement('span');label.textContent=r.label;const track=document.createElement('span');track.className='bar-track';const b=document.createElement('i');b.style.width=`${Math.max(0,Math.min(100,r.v))}%`;track.append(b);const val=document.createElement('strong');val.textContent=`${num(r.v)} %`;el.append(label,track,val);return el;}));}
async function verify(){const {digest,...rest}=report;const value=JSON.stringify(rest);let hash;if(globalThis.crypto?.subtle){const raw=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value));hash=[...new Uint8Array(raw)].map(x=>x.toString(16).padStart(2,'0')).join('');}else{hash=sha256(value);}const ok=hash===digest;$('verification').textContent=ok?'PASS · empreinte du rapport SHA-256 vérifiée':'FAIL · données archivées non conformes';$('digest').textContent=`SHA-256 : ${hash}`;return ok;}
function replaySeed(){if(!report)return;const scenario=$('scenario').value,seed=Number($('seed').value),r=report.rows.find(x=>x.case===scenario&&x.seed===seed);if(!r)return;
  lastEvidence={schema:'astra-mars-replay/1',sourceDigest:report.digest,original:r,warning:'Relecture de résultats synthétiques archivés, sans réexécution du moteur de calcul'};
  $('base').textContent=`${num(r.baseline.essentialServicePct)} %`;$('react').textContent=`${num(r.reactive.essentialServicePct)} %`;$('pred').textContent=`${num(r.risk.essentialServicePct)} %`;
  colorDelta($('trade'),r.nominalEquivalentDeltaPctPoints);colorDelta($('delta'),r.predictiveGainVsReactivePts);colorDelta($('nominal'),r.nominalEquivalentDeltaPctPoints);
  $('count').textContent='3';$('compare-status').textContent=`REPLAY · ${cases[scenario]} · graine ${seed} · 240 h archivées`;
  renderBars([{label:'Fixe',v:r.baseline.essentialServicePct},{label:'Réactif',v:r.reactive.essentialServicePct},{label:'ASTRA',v:r.risk.essentialServicePct}]);$('warning').textContent=`Fonctions essentielles : fixe ${num(r.baseline.essentialServicePct)} %, réactif ${num(r.reactive.essentialServicePct)} %, ASTRA ${num(r.risk.essentialServicePct)} %. Compromis nominal ${num(r.nominalEquivalentDeltaPctPoints)} point(s). Modèle non certifié.`;
}
function replayAll(){if(!report)return;const c=report.byCase.find(x=>x.case==='combined');colorDelta($('delta'),c.avgPredictiveGainVsReactivePts);colorDelta($('nominal'),c.avgNominalCapacityDeltaPctPoints);$('count').textContent=String(report.rows.length*3);
  $('compare-status').textContent=`REPLAY · ${report.byCase.length} scénarios, ${report.seeds} graines, ${report.rows.length*3} trajectoires historiques`;
  renderBars(report.byCase.map(x=>({label:cases[x.case],v:x.avgRiskServicePct})));lastEvidence=report;
  $('warning').textContent=`Contrôles négatifs conservés : ${report.byCase.filter(x=>x.avgPredictiveGainVsReactivePts<=0).map(x=>cases[x.case]).join(', ')||'aucun'}. Détail complet dans l’export.`;
}
function download(){if(!lastEvidence){$('verification').textContent='Choisir un scénario avant export';return;}const b=new Blob([JSON.stringify(lastEvidence,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='ASTRA_MARS_OS_ARCHIVED_REPLAY.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),4000);}
function scene(){const c=$('terrain'),r=c.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=r.width,h=r.height;if(!w||!h)return;c.width=w*ratio;c.height=h*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);const p=(x,y,z)=>{const a=Math.cos(angle),b=Math.sin(angle),rx=a*x+b*z,rz=-b*x+a*z,s=470/(640+rz);return [w/2+rx*s,h*.69+rz*.27-y*s]};
 ctx.fillStyle='#ad6545';ctx.globalAlpha=.22;for(let i=0;i<100;i++){const x=Math.sin(i*41.1)*370,z=Math.cos(i*9.5)*330;const [xx,yy]=p(x,0,z);ctx.fillRect(xx,yy,2,2)}ctx.globalAlpha=1;
 for(let t=-280;t<=280;t+=65){ctx.beginPath();for(let z=-320;z<=320;z+=22){const [x,y]=p(t,0,z);z===-320?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.strokeStyle='rgba(245,172,123,.12)';ctx.stroke();}
 const boxes=[[-130,-40,105,75,42,'#fcb486','HAB-01'],[55,-80,80,58,35,'#fcb486','HAB-02'],[-5,110,90,60,20,'#7de6b8','AGRI'],[180,60,95,35,7,'#6ae7ed','SOLAR'],[-185,118,80,33,7,'#6ae7ed','SOLAR'],[155,-90,25,25,48,'#f1d398','PWR']];
 const polygon=(a,clr,alpha=.25)=>{ctx.beginPath();a.forEach((point,i)=>i?ctx.lineTo(...point):ctx.moveTo(...point));ctx.closePath();ctx.fillStyle=clr;ctx.globalAlpha=alpha;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=clr;ctx.stroke()};
 for(const [x,z,wi,de,he,clr,name] of boxes){const q=[p(x-wi/2,0,z-de/2),p(x+wi/2,0,z-de/2),p(x+wi/2,0,z+de/2),p(x-wi/2,0,z+de/2)],top=q.map((_,i)=>p(x+(i===1||i===2?wi/2:-wi/2),he,z+(i>=2?de/2:-de/2)));polygon([q[1],q[2],top[2],top[1]],clr,.25);polygon([q[0],q[1],top[1],top[0]],clr,.17);polygon(top,clr,.3);const [xx,yy]=p(x,he+20,z);ctx.fillStyle='#e7f5ff';ctx.textAlign='center';ctx.font='bold 11px system-ui';ctx.fillText(name,xx,yy)}
 for(let i=0;i<4;i++){const [x,y]=p(-155+i*91,7,170-i*35);ctx.fillStyle='#90f5c4';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#d5fff0';ctx.font='10px system-ui';ctx.fillText(`ANT-${i+1}`,x,y-13)}
}

$('run').addEventListener('click',replaySeed);$('bench').addEventListener('click',replayAll);$('export').addEventListener('click',download);
$('terrain').addEventListener('pointerdown',e=>{dragging=true;px=e.clientX;$('terrain').setPointerCapture(e.pointerId)});
$('terrain').addEventListener('pointermove',e=>{if(!dragging)return;angle+=(e.clientX-px)*.009;px=e.clientX;scene()});
$('terrain').addEventListener('pointerup',()=>dragging=false);$('terrain').addEventListener('pointercancel',()=>dragging=false);
window.addEventListener('resize',scene);
scene();fetch('/data.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.json()}).then(async r=>{report=r;const good=await verify();$('health').textContent=good?`● Archives ${r.version} vérifiées · 420 trajectoires`:'FAIL · intégrité';replaySeed()}).catch(e=>{$('health').textContent=`FAIL · ${e.message}`;$('verification').textContent='FAIL · Données indisponibles'});