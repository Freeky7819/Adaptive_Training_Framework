import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Play, Pause, Square, Settings, Zap, Brain, Activity, TrendingUp, Sliders, BarChart2, Target, Cpu, RefreshCw, Download, ChevronDown, ChevronRight, Check, Sparkles } from 'lucide-react';

const DEFAULT_CONFIG = {
  dataset: 'cifar10', model: 'cifar10cnn', epochs: 50, batchSize: 128, learningRate: 0.001,
  useConvergenceAnalysis: true, useGradientFeedback: true, usePeriodicLR: true,
  useConvergenceDamper: true, useTemporalBuffer: true, useHarmonicInit: true,
  useCurvatureRegularizer: false, useMetaController: true,
  lrOmega: 6.0, lrAmplitude: 0.08, lrDecay: 0.003, lrPhase: 1.0472,
  gfAlpha: 0.08, gfOmega: 6.0, cdThreshold: 0.008, cdAlphaDamp: 0.40,
};

const DATASETS = [
  { value: 'mnist', label: 'MNIST', classes: 10 },
  { value: 'fashion_mnist', label: 'Fashion-MNIST', classes: 10 },
  { value: 'cifar10', label: 'CIFAR-10', classes: 10 },
  { value: 'cifar100', label: 'CIFAR-100', classes: 100 },
  { value: 'svhn', label: 'SVHN', classes: 10 },
  { value: 'bert_sst2', label: 'GLUE SST-2', classes: 2 },
];

const Toggle = ({ enabled, onChange, label, description }) => (
  <div className="flex items-center justify-between py-1.5">
    <div className="flex-1">
      <div className="text-xs font-medium text-gray-200">{label}</div>
      {description && <div className="text-[10px] text-gray-500">{description}</div>}
    </div>
    <button onClick={() => onChange(!enabled)}
      className={`relative w-9 h-5 rounded-full transition-colors ${enabled ? 'bg-blue-600' : 'bg-gray-600'}`}>
      <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${enabled ? 'translate-x-4' : ''}`} />
    </button>
  </div>
);

const ParamSlider = ({ label, value, onChange, min, max, step }) => (
  <div className="py-1">
    <div className="flex justify-between items-center mb-0.5">
      <span className="text-xs text-gray-300">{label}</span>
      <input type="number" value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-16 px-1 py-0.5 text-xs bg-gray-700 border border-gray-600 rounded text-right text-white"
        step={step} min={min} max={max} />
    </div>
    <input type="range" value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
      min={min} max={max} step={step}
      className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
  </div>
);

const Section = ({ title, icon: Icon, children, open = true }) => {
  const [isOpen, setIsOpen] = useState(open);
  return (
    <div className="border border-gray-700 rounded mb-2 overflow-hidden">
      <button onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-1.5 px-2 py-1.5 bg-gray-800 hover:bg-gray-750">
        {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {Icon && <Icon size={12} className="text-blue-400" />}
        <span className="text-xs font-medium">{title}</span>
      </button>
      {isOpen && <div className="px-2 py-1.5 bg-gray-850">{children}</div>}
    </div>
  );
};

const MetricCard = ({ label, value, unit = '', color = 'blue' }) => {
  const colors = { blue: 'border-blue-500/30', green: 'border-green-500/30', purple: 'border-purple-500/30', orange: 'border-orange-500/30' };
  return (
    <div className={`bg-gray-800 border ${colors[color]} rounded p-2`}>
      <div className="text-[10px] text-gray-400">{label}</div>
      <div className="flex items-baseline gap-0.5">
        <span className="text-lg font-bold text-white">{value}</span>
        <span className="text-[10px] text-gray-500">{unit}</span>
      </div>
    </div>
  );
};

export default function ATFDashboard() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [status, setStatus] = useState('idle');
  const [epoch, setEpoch] = useState(0);
  const [metrics, setMetrics] = useState({ trainLoss: [], valLoss: [], accuracy: [], lr: [], beta: [] });
  const [moduleStats, setModuleStats] = useState({ gf: 0, cd: 0, lr: 0.001, mc: 0 });
  const [best, setBest] = useState({ acc: 0, loss: 999, epoch: 0 });
  const [logs, setLogs] = useState([]);
  const [tab, setTab] = useState('train');
  const [tunerResults, setTunerResults] = useState([]);
  const [tunerBest, setTunerBest] = useState(null);
  const [tuning, setTuning] = useState(false);
  const intervalRef = useRef(null);

  const log = (msg, type = 'info') => {
    setLogs(p => [...p.slice(-49), { t: new Date().toLocaleTimeString(), msg, type }]);
  };

  const simulate = useCallback(() => {
    setEpoch(e => {
      const newE = e + 1;
      const base = 2.5 * Math.exp(-0.03 * newE) + 0.3;
      const noise = (Math.random() - 0.5) * 0.1;
      const pmod = config.usePeriodicLR ? 0.02 * Math.sin(config.lrOmega * Math.log(1 + newE)) : 0;
      const tLoss = Math.max(0.1, base + noise + pmod);
      const vLoss = Math.max(0.15, base * 1.1 + noise * 0.5);
      const acc = Math.min(99, 50 + 45 * (1 - Math.exp(-0.05 * newE)) + Math.random() * 2);
      const baseLR = config.learningRate * Math.exp(-config.lrDecay * newE);
      const lr = config.usePeriodicLR ? baseLR * (1 + config.lrAmplitude * Math.sin(config.lrOmega * Math.log(1 + newE))) : baseLR;
      const beta = Math.max(0.001, 0.1 * Math.exp(-0.08 * newE) + Math.random() * 0.01);

      setMetrics(p => ({
        trainLoss: [...p.trainLoss, { e: newE, v: tLoss }],
        valLoss: [...p.valLoss, { e: newE, v: vLoss }],
        accuracy: [...p.accuracy, { e: newE, v: acc }],
        lr: [...p.lr, { e: newE, v: lr }],
        beta: [...p.beta, { e: newE, v: beta }],
      }));

      setModuleStats({ gf: config.useGradientFeedback ? Math.random() * 0.05 : 0, cd: config.useConvergenceDamper && beta < config.cdThreshold ? 1 : 0, lr, mc: newE % 15 === 0 ? 1 : 0 });

      if (acc > best.acc) {
        setBest({ acc, loss: vLoss, epoch: newE });
        log(`★ New best: ${acc.toFixed(2)}%`, 'success');
      }

      if (newE >= config.epochs) {
        setStatus('done');
        log('Training complete!', 'success');
        clearInterval(intervalRef.current);
      }
      return newE;
    });
  }, [config, best.acc]);

  const start = () => {
    if (status === 'running') return;
    setStatus('running');
    log('Started');
    if (epoch === 0) {
      setMetrics({ trainLoss: [], valLoss: [], accuracy: [], lr: [], beta: [] });
      setBest({ acc: 0, loss: 999, epoch: 0 });
    }
    intervalRef.current = setInterval(simulate, 200);
  };

  const pause = () => { setStatus('paused'); log('Paused', 'warn'); clearInterval(intervalRef.current); };
  const stop = () => { setStatus('idle'); setEpoch(0); log('Stopped'); clearInterval(intervalRef.current); };

  const runTuner = async () => {
    setTuning(true); setTunerResults([]); log('Auto-tuner started');
    const results = [];
    for (let i = 0; i < 10; i++) {
      const cfg = { ...config, lrOmega: 4 + Math.random() * 4, lrAmplitude: 0.04 + Math.random() * 0.08, learningRate: 0.0005 + Math.random() * 0.002 };
      const base = 75 + Math.random() * 10;
      const bonus = Math.abs(cfg.lrOmega - 6) < 1 ? 3 : 0;
      const acc = base + bonus + Math.random() * 2;
      results.push({ id: i + 1, cfg, acc, loss: 2.5 - acc / 50 });
      setTunerResults([...results]);
      await new Promise(r => setTimeout(r, 300));
    }
    const b = results.reduce((a, c) => a.acc > c.acc ? a : c);
    setTunerBest(b);
    log(`Best: ${b.acc.toFixed(2)}% (ω=${b.cfg.lrOmega.toFixed(2)})`, 'success');
    setTuning(false);
  };

  const applyBest = () => { if (tunerBest) { setConfig(tunerBest.cfg); log('Applied best config'); } };
  const exportCfg = () => { const b = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' }); const u = URL.createObjectURL(b); const a = document.createElement('a'); a.href = u; a.download = 'atf_config.json'; a.click(); };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  const chartData = metrics.trainLoss.map((x, i) => ({ e: x.e, tL: x.v, vL: metrics.valLoss[i]?.v, acc: metrics.accuracy[i]?.v, lr: metrics.lr[i]?.v * 10000, beta: metrics.beta[i]?.v * 100 }));

  return (
    <div className="min-h-screen bg-gray-900 text-white text-sm">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-3 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded flex items-center justify-center"><Brain size={18} /></div>
          <div><div className="font-bold text-sm">ATF Dashboard</div><div className="text-[10px] text-gray-400">Adaptive Training Framework</div></div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${status === 'running' ? 'bg-green-600 animate-pulse' : status === 'paused' ? 'bg-yellow-600' : status === 'done' ? 'bg-blue-600' : 'bg-gray-600'}`}>{status.toUpperCase()}</span>
          <button onClick={exportCfg} className="p-1.5 hover:bg-gray-700 rounded" title="Export"><Download size={14} /></button>
        </div>
      </header>

      <div className="flex" style={{ height: 'calc(100vh - 48px)' }}>
        {/* Sidebar */}
        <aside className="w-64 bg-gray-850 border-r border-gray-700 overflow-y-auto p-2">
          <Section title="Dataset & Model" icon={Cpu}>
            <select value={config.dataset} onChange={e => setConfig(p => ({ ...p, dataset: e.target.value }))}
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs mb-1" disabled={status === 'running'}>
              {DATASETS.map(d => <option key={d.value} value={d.value}>{d.label} ({d.classes})</option>)}
            </select>
          </Section>

          <Section title="Training" icon={Settings}>
            <ParamSlider label="Epochs" value={config.epochs} onChange={v => setConfig(p => ({ ...p, epochs: v }))} min={1} max={200} step={1} />
            <ParamSlider label="Batch Size" value={config.batchSize} onChange={v => setConfig(p => ({ ...p, batchSize: v }))} min={16} max={512} step={16} />
            <ParamSlider label="Learning Rate" value={config.learningRate} onChange={v => setConfig(p => ({ ...p, learningRate: v }))} min={0.0001} max={0.01} step={0.0001} />
          </Section>

          <Section title="ATF Modules" icon={Zap}>
            <Toggle label="Convergence Analysis" enabled={config.useConvergenceAnalysis} onChange={v => setConfig(p => ({ ...p, useConvergenceAnalysis: v }))} description="Early stopping" />
            <Toggle label="Gradient Feedback" enabled={config.useGradientFeedback} onChange={v => setConfig(p => ({ ...p, useGradientFeedback: v }))} description="Loss modulation" />
            <Toggle label="Periodic LR" enabled={config.usePeriodicLR} onChange={v => setConfig(p => ({ ...p, usePeriodicLR: v }))} description="Log-periodic oscillation" />
            <Toggle label="Convergence Damper" enabled={config.useConvergenceDamper} onChange={v => setConfig(p => ({ ...p, useConvergenceDamper: v }))} description="LR reduction" />
            <Toggle label="Temporal Buffer" enabled={config.useTemporalBuffer} onChange={v => setConfig(p => ({ ...p, useTemporalBuffer: v }))} description="History adaptation" />
            <Toggle label="Harmonic Init" enabled={config.useHarmonicInit} onChange={v => setConfig(p => ({ ...p, useHarmonicInit: v }))} description="Weight perturbation" />
            <Toggle label="Meta Controller" enabled={config.useMetaController} onChange={v => setConfig(p => ({ ...p, useMetaController: v }))} description="Epoch adaptation" />
          </Section>

          {config.usePeriodicLR && (
            <Section title="Periodic LR Params" icon={Activity} open={false}>
              <ParamSlider label="ω (omega)" value={config.lrOmega} onChange={v => setConfig(p => ({ ...p, lrOmega: v }))} min={1} max={12} step={0.1} />
              <ParamSlider label="α (amplitude)" value={config.lrAmplitude} onChange={v => setConfig(p => ({ ...p, lrAmplitude: v }))} min={0} max={0.2} step={0.01} />
              <ParamSlider label="k (decay)" value={config.lrDecay} onChange={v => setConfig(p => ({ ...p, lrDecay: v }))} min={0} max={0.01} step={0.0001} />
              <div className="text-[10px] text-gray-500 mt-1">Universal frequency ω ≈ 6.0</div>
            </Section>
          )}

          {config.useConvergenceDamper && (
            <Section title="Damper Params" icon={Target} open={false}>
              <ParamSlider label="β threshold" value={config.cdThreshold} onChange={v => setConfig(p => ({ ...p, cdThreshold: v }))} min={0.001} max={0.02} step={0.001} />
              <ParamSlider label="α_damp" value={config.cdAlphaDamp} onChange={v => setConfig(p => ({ ...p, cdAlphaDamp: v }))} min={0.1} max={0.8} step={0.05} />
            </Section>
          )}
        </aside>

        {/* Main */}
        <main className="flex-1 overflow-y-auto">
          {/* Controls */}
          <div className="bg-gray-800 border-b border-gray-700 px-3 py-2 flex items-center justify-between">
            <div className="flex gap-1">
              {status !== 'running' ? (
                <button onClick={start} disabled={tuning} className="flex items-center gap-1 px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-xs font-medium">
                  <Play size={12} /> {epoch > 0 ? 'Resume' : 'Start'}
                </button>
              ) : (
                <button onClick={pause} className="flex items-center gap-1 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs font-medium">
                  <Pause size={12} /> Pause
                </button>
              )}
              <button onClick={stop} disabled={status === 'idle' || tuning} className="flex items-center gap-1 px-3 py-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded text-xs font-medium">
                <Square size={12} /> Stop
              </button>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs"><span className="text-gray-400">Epoch:</span> <span className="font-mono font-bold">{epoch}/{config.epochs}</span></span>
              <div className="w-32 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 transition-all" style={{ width: `${(epoch / config.epochs) * 100}%` }} />
              </div>
            </div>
            <div className="flex gap-0.5 bg-gray-700 rounded p-0.5">
              {['train', 'modules', 'tuner'].map(t => (
                <button key={t} onClick={() => setTab(t)}
                  className={`px-2 py-0.5 rounded text-xs ${tab === t ? 'bg-blue-600' : 'text-gray-400 hover:text-white'}`}>
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="p-3">
            {tab === 'train' && (
              <>
                {/* Metrics */}
                <div className="grid grid-cols-4 gap-2 mb-3">
                  <MetricCard label="Best Accuracy" value={best.acc.toFixed(2)} unit="%" color="green" />
                  <MetricCard label="Current Loss" value={metrics.trainLoss.slice(-1)[0]?.v.toFixed(4) || '—'} color="blue" />
                  <MetricCard label="Learning Rate" value={(moduleStats.lr * 1000).toFixed(3)} unit="×10⁻³" color="purple" />
                  <MetricCard label="Beta (β)" value={(metrics.beta.slice(-1)[0]?.v * 100 || 0).toFixed(2)} unit="×10⁻²" color="orange" />
                </div>

                {/* Charts */}
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div className="bg-gray-800 border border-gray-700 rounded p-2">
                    <div className="text-xs font-medium mb-2 flex items-center gap-1"><BarChart2 size={12} className="text-blue-400" /> Loss</div>
                    <ResponsiveContainer width="100%" height={140}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="e" stroke="#9CA3AF" fontSize={9} />
                        <YAxis stroke="#9CA3AF" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', fontSize: 10 }} />
                        <Line type="monotone" dataKey="tL" stroke="#3B82F6" name="Train" dot={false} strokeWidth={1.5} />
                        <Line type="monotone" dataKey="vL" stroke="#F59E0B" name="Val" dot={false} strokeWidth={1.5} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-gray-800 border border-gray-700 rounded p-2">
                    <div className="text-xs font-medium mb-2 flex items-center gap-1"><Target size={12} className="text-green-400" /> Accuracy</div>
                    <ResponsiveContainer width="100%" height={140}>
                      <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="e" stroke="#9CA3AF" fontSize={9} />
                        <YAxis domain={[0, 100]} stroke="#9CA3AF" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', fontSize: 10 }} />
                        <Area type="monotone" dataKey="acc" stroke="#10B981" fill="#10B981" fillOpacity={0.3} name="Acc%" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-gray-800 border border-gray-700 rounded p-2">
                    <div className="text-xs font-medium mb-2 flex items-center gap-1"><Activity size={12} className="text-purple-400" /> Learning Rate</div>
                    <ResponsiveContainer width="100%" height={140}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="e" stroke="#9CA3AF" fontSize={9} />
                        <YAxis stroke="#9CA3AF" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', fontSize: 10 }} />
                        <Line type="monotone" dataKey="lr" stroke="#A855F7" name="LR×10⁴" dot={false} strokeWidth={1.5} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="bg-gray-800 border border-gray-700 rounded p-2">
                    <div className="text-xs font-medium mb-2 flex items-center gap-1"><TrendingUp size={12} className="text-orange-400" /> Beta (Convergence)</div>
                    <ResponsiveContainer width="100%" height={140}>
                      <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="e" stroke="#9CA3AF" fontSize={9} />
                        <YAxis stroke="#9CA3AF" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', fontSize: 10 }} />
                        <Area type="monotone" dataKey="beta" stroke="#F97316" fill="#F97316" fillOpacity={0.3} name="β×10²" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Logs */}
                <div className="bg-gray-800 border border-gray-700 rounded p-2">
                  <div className="text-xs font-medium mb-1">Log</div>
                  <div className="h-20 overflow-y-auto font-mono text-[10px] bg-gray-900 rounded p-1.5">
                    {logs.map((l, i) => (
                      <div key={i} className={l.type === 'success' ? 'text-green-400' : l.type === 'warn' ? 'text-yellow-400' : 'text-gray-400'}>
                        <span className="text-gray-600">[{l.t}]</span> {l.msg}
                      </div>
                    ))}
                    {logs.length === 0 && <div className="text-gray-600">Ready...</div>}
                  </div>
                </div>
              </>
            )}

            {tab === 'modules' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { name: 'Gradient Feedback', on: config.useGradientFeedback, icon: Zap, color: 'yellow', stat: `Gain: ${(moduleStats.gf * 100).toFixed(1)}%` },
                    { name: 'Convergence Damper', on: config.useConvergenceDamper, icon: Target, color: 'blue', stat: `Active: ${moduleStats.cd ? 'Yes' : 'No'}` },
                    { name: 'Periodic LR', on: config.usePeriodicLR, icon: Activity, color: 'purple', stat: `LR: ${moduleStats.lr.toExponential(2)}` },
                    { name: 'Meta Controller', on: config.useMetaController, icon: Brain, color: 'green', stat: `Reductions: ${moduleStats.mc}` },
                  ].map(m => (
                    <div key={m.name} className="bg-gray-800 border border-gray-700 rounded p-2">
                      <div className="flex items-center gap-1.5 mb-1">
                        <m.icon size={12} className={`text-${m.color}-400`} />
                        <span className="text-xs font-medium">{m.name}</span>
                        <span className={`ml-auto text-[10px] ${m.on ? 'text-green-400' : 'text-gray-500'}`}>{m.on ? 'ON' : 'OFF'}</span>
                      </div>
                      <div className="text-[10px] text-gray-400">{m.stat}</div>
                    </div>
                  ))}
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded p-2">
                  <div className="text-xs font-medium mb-2">Active Formulas</div>
                  <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                    {config.usePeriodicLR && (
                      <div className="bg-gray-900 rounded p-2">
                        <div className="text-purple-400">η(t) = η₀·e⁻ᵏᵗ·(1 + α·sin(ω·log(1+t)))</div>
                        <div className="text-gray-500">ω={config.lrOmega}, α={config.lrAmplitude}</div>
                      </div>
                    )}
                    {config.useConvergenceDamper && (
                      <div className="bg-gray-900 rounded p-2">
                        <div className="text-blue-400">lr' = lr·(1 - α_damp·e⁻ᵝ²/ε)</div>
                        <div className="text-gray-500">β_th={config.cdThreshold}</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {tab === 'tuner' && (
              <div className="space-y-3">
                <div className="bg-gray-800 border border-gray-700 rounded p-3">
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles size={14} className="text-purple-400" />
                    <span className="text-xs font-medium">Auto-Tuner</span>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={runTuner} disabled={status === 'running' || tuning}
                      className="flex items-center gap-1 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded text-xs font-medium">
                      <RefreshCw size={12} className={tuning ? 'animate-spin' : ''} />
                      {tuning ? 'Running 10 trials...' : 'Run 10 Trials'}
                    </button>
                    {tunerBest && (
                      <button onClick={applyBest} className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded text-xs font-medium">
                        <Check size={12} /> Apply Best
                      </button>
                    )}
                  </div>
                </div>

                {tunerResults.length > 0 && (
                  <div className="bg-gray-800 border border-gray-700 rounded p-2 overflow-x-auto">
                    <table className="w-full text-[10px]">
                      <thead><tr className="text-gray-400 border-b border-gray-700">
                        <th className="text-left py-1 px-1">#</th><th className="text-left py-1 px-1">ω</th>
                        <th className="text-left py-1 px-1">α</th><th className="text-left py-1 px-1">LR</th>
                        <th className="text-left py-1 px-1">Accuracy</th>
                      </tr></thead>
                      <tbody>
                        {tunerResults.map(r => (
                          <tr key={r.id} className={`border-b border-gray-700/50 ${tunerBest?.id === r.id ? 'bg-green-900/30' : ''}`}>
                            <td className="py-1 px-1">{r.id}</td>
                            <td className="py-1 px-1 font-mono">{r.cfg.lrOmega.toFixed(2)}</td>
                            <td className="py-1 px-1 font-mono">{r.cfg.lrAmplitude.toFixed(3)}</td>
                            <td className="py-1 px-1 font-mono">{r.cfg.learningRate.toFixed(4)}</td>
                            <td className="py-1 px-1 font-mono text-green-400">{r.acc.toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {tunerBest && (
                  <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-700/50 rounded p-3">
                    <div className="text-xs font-medium text-green-400 mb-2 flex items-center gap-1">
                      <Sparkles size={12} /> Best Configuration
                    </div>
                    <div className="grid grid-cols-4 gap-3">
                      <div><div className="text-[10px] text-gray-400">Omega</div><div className="text-lg font-bold font-mono">{tunerBest.cfg.lrOmega.toFixed(2)}</div></div>
                      <div><div className="text-[10px] text-gray-400">Amplitude</div><div className="text-lg font-bold font-mono">{tunerBest.cfg.lrAmplitude.toFixed(3)}</div></div>
                      <div><div className="text-[10px] text-gray-400">LR</div><div className="text-lg font-bold font-mono">{tunerBest.cfg.learningRate.toFixed(4)}</div></div>
                      <div><div className="text-[10px] text-gray-400">Accuracy</div><div className="text-lg font-bold font-mono text-green-400">{tunerBest.acc.toFixed(2)}%</div></div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
