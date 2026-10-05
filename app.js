var motifs=['Diamond','Zigzag','Floral'];
var pals=[{n:'Crimson',bg:'#8E1B2E',a:'#E8B84A',b:'#F6E3B0'},{n:'Teal',bg:'#0F6E56',a:'#F4C0D1',b:'#FAECE7'},{n:'Indigo',bg:'#26215C',a:'#CECBF6',b:'#F1EFE8'},{n:'Forest',bg:'#27500A',a:'#EF9F27',b:'#FAEEDA'}];
var S={m:0,p:0,mode:1};
function shape(m,p){
 if(m==0)return '<polygon points="12,2 22,12 12,22 2,12" fill="'+p.a+'"/><circle cx="12" cy="12" r="3" fill="'+p.b+'"/>';
 if(m==1)return '<path d="M0 10 L6 4 L12 10 L18 4 L24 10" stroke="'+p.a+'" stroke-width="3" fill="none"/><circle cx="12" cy="19" r="2.5" fill="'+p.b+'"/>';
 return '<circle cx="12" cy="12" r="4.5" fill="'+p.a+'"/><circle cx="12" cy="3" r="2" fill="'+p.b+'"/><circle cx="12" cy="21" r="2" fill="'+p.b+'"/><circle cx="3" cy="12" r="2" fill="'+p.b+'"/><circle cx="21" cy="12" r="2" fill="'+p.b+'"/>';
}
function tile(m,pi,id){
 var p=pals[pi];
 return '<svg viewBox="0 0 96 96" role="img" aria-label="'+motifs[m]+' motif in '+p.n+'"><defs><pattern id="'+id+'" width="24" height="24" patternUnits="userSpaceOnUse"><rect width="24" height="24" fill="'+p.bg+'"/>'+shape(m,p)+'</pattern></defs><rect width="96" height="96" fill="url(#'+id+')"/></svg>';
}
function h(a,b){var x=Math.sin(a*12.9898+b*78.233)*43758.5453;return x-Math.floor(x);}
function sim(qm,qp,m,p){
 var j=h(qm*10+qp,m*10+p),sm=qm==m,sp=qp==p;
 if(S.mode==1)return sm?0.78+0.14*j:0.04+0.3*j;
 return 0.15+(sp?0.45:0)+(sm?0.2:0)+0.15*j;
}
function seg(id,items,key){
 var el=document.getElementById(id);el.innerHTML='';
 items.forEach(function(t,i){var b=document.createElement('button');b.textContent=t;b.className=(S[key]==i)?'on':'';b.onclick=function(){S[key]=i;render();};el.appendChild(b);});
}
function render(){
 seg('mSeg',motifs,'m');seg('pSeg',pals.map(function(p){return p.n}),'p');
 seg('modeSeg',['Colour-sensitive (untrained)','Colour-invariant (your model)'],'mode');
 var thr=parseFloat(document.getElementById('thr').value);
 document.getElementById('thrOut').textContent=thr.toFixed(2);
 document.getElementById('q').innerHTML=tile(S.m,S.p,'pq')+'<p style="font-size:12px;margin:4px 0 0">'+motifs[S.m]+', '+pals[S.p].n+'</p>';
 var items=[];
 for(var m=0;m<3;m++)for(var p=0;p<4;p++){if(m==S.m&&p==S.p)continue;items.push({m:m,p:p,s:sim(S.m,S.p,m,p)});}
 items.sort(function(a,b){return b.s-a.s});
 var g=document.getElementById('grid');g.innerHTML='';
 var top3=0,fp=0;
 items.forEach(function(it,i){
  var same=it.m==S.m,pred=it.s>=thr;
  if(i<3&&same)top3++;
  if(pred&&!same)fp++;
  var d=document.createElement('div');d.className='tile '+(same?'ok':'bad');
  d.innerHTML=tile(it.m,it.p,'g'+i)+'<div class="bar"><i style="width:'+Math.round(it.s*100)+'%"></i></div><div style="display:flex;justify-content:space-between;font-size:11px"><span>#'+(i+1)+'  '+it.s.toFixed(2)+'</span><span style="font-weight:500;color:var(--'+(pred?'text-success':'text-secondary')+')">'+(pred?'SAME':'diff')+'</span></div><div style="font-size:11px;color:var(--text-secondary)">'+motifs[it.m]+', '+pals[it.p].n+'</div>';
  g.appendChild(d);
 });
 document.getElementById('sum').textContent='Green border = truly the same design (3 colorways exist in the gallery). Correct designs in the top 3: '+top3+' of 3. Wrong designs accepted as SAME at this threshold: '+fp+'.';
}
document.getElementById('thr').oninput=render;
render();
