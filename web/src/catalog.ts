import raw from '../../src/fieldsim/catalog.json';
import type { FieldDescriptor, Model, Parameters, ParameterKey, Preset, Setup, TileConfig } from './contracts';
export const catalog = raw;
export const presets = [...raw.presets as unknown as Preset[]].sort((a,b) => Number(b.model === 'ecology') - Number(a.model === 'ecology'));
export interface ParameterDefinition { key: ParameterKey; label: string; default: number; min: number; max: number; step: number; group?: string; }
export const parameterDefinitions = raw.parameters as ParameterDefinition[];
export function getParameterDefinitions(model: Model): ParameterDefinition[] {
  if (model !== 'ecology') return parameterDefinitions;
  const definitions = (raw as unknown as { ecologyParameters: ParameterDefinition[] }).ecologyParameters;
  const groups: Record<string,string> = {dp:'Settlement',df:'Food and cultivation',dw:'Water supply and demand',chiFood:'Settlement',chiWater:'Settlement',growth:'Settlement',yield:'Food and cultivation',cultivationScale:'Food and cultivation',waterConsumption:'Water supply and demand',harvestWaterCost:'Water supply and demand',consumption:'Food and cultivation',spoilage:'Food and cultivation',foodSupport:'Settlement',waterSupport:'Settlement',replenishmentRate:'Water supply and demand',sourceCapacity:'Water supply and demand',soilRecovery:'Soil depletion and recovery',erosion:'Soil depletion and recovery',settlementErosion:'Soil depletion and recovery'};
  const order = ['Settlement','Water supply and demand','Food and cultivation','Soil depletion and recovery'];
  return order.flatMap(group => definitions.filter(d => groups[d.key] === group).map(d => ({...d,group})));
}
export function getPreset(id: string): Preset { const p = presets.find(p => p.id === id); if (!p) throw new Error(`Unknown experiment: ${id}`); return p; }
const descriptors: FieldDescriptor[] = [
  {key:'population',label:'Population',palette:[[22,48,43],[210,101,55],[255,218,148]],scale:2,color:'#d96139',editable:true,kind:'dynamic'},
  {key:'food',label:'Food stock',palette:[[24,48,42],[205,170,76],[255,235,164]],scale:3,color:'#b58b20',editable:true,kind:'dynamic'},
  {key:'infrastructure',label:'Infrastructure',palette:[[19,42,43],[86,166,137],[207,239,187]],scale:0.5,color:'#3c9472',editable:true,kind:'dynamic'},
  {key:'soil',label:'Soil condition',palette:[[49,36,31],[181,128,82],[226,211,161]],scale:1,color:'#9d654c',bounded:[0,1],editable:true,kind:'dynamic'},
  {key:'fertility',label:'Fertility',palette:[[24,49,38],[151,173,83],[230,230,162]],scale:4,color:'#6c984a',editable:true,kind:'static'},
  {key:'water',label:'Water',palette:[[12,35,54],[39,135,177],[176,232,244]],scale:2,color:'#287fa8',editable:true,kind:'dynamic'},
  {key:'waterSources',label:'Water sources',palette:[[21,40,55],[64,112,171],[204,217,255]],scale:1,color:'#6577bd',bounded:[0,1],editable:true,kind:'static'},
  {key:'cultivation',label:'Cultivation',palette:[[35,40,25],[162,148,46],[244,230,126]],scale:1,color:'#91832a',bounded:[0,1],editable:false,kind:'derived'},
];
export function getFieldDescriptors(model: Model): FieldDescriptor[] {
  return descriptors.filter(field => model === 'ecology' ? field.key !== 'infrastructure' : !['water','waterSources','cultivation'].includes(field.key)).map(field => ({...field,kind: (model === 'agriculture' || model === 'chemotaxis') && ['soil','infrastructure'].includes(field.key) ? 'static' : field.kind}));
}
export function defaultTiles(model: Model): TileConfig[] {
  return (model === 'ecology' ? ['population','water','soil'] as const : ['population'] as const).map((field,index) => ({id:`view-${index+1}`,layers:[{field,opacity:1,visible:true}],paintField:field}));
}
export function defaultSetup(id = 'settlement'): Setup {
  const preset = getPreset(id);
  return {version:preset.model === 'ecology' ? 2 : 1,preset:id,seed:0,n:64,boundary:'neumann',parameters:{...preset.parameters} as Parameters,...(preset.model === 'ecology' ? {tiles:defaultTiles(preset.model)} : {})};
}
