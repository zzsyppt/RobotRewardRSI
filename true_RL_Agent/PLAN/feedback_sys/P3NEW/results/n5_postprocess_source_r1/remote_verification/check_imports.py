from pathlib import Path
import importlib,json,sys
root=Path(sys.argv[1]).resolve()
names=('true_rl_agent','prepare_confirmation_materials','prepare_confirmation_media',
       'prepare_native_observer','prepare_text_author_smoke','prepare_native_author_smoke',
       'prepare_native_author_final_slots','audit_delivery_offline',
       'confirmation_context','author_ceiling_probe','author_experiment')
modules={name:importlib.import_module(name) for name in names}
modules['confirmation_context._p3']=modules['confirmation_context']._p3(root)
modules['author_ceiling_probe.load_p3']=modules['author_ceiling_probe'].load_p3(root/'PLAN/feedback_sys/P3')
paths={name:str(Path(module.__file__).resolve()) for name,module in modules.items()}
assert all(Path(path).is_relative_to(root) for path in paths.values()),paths
for name in ('prepare_native_author_smoke','prepare_native_author_final_slots','prepare_text_author_smoke'):
 module=modules[name]
 for attr in ('ROOT','SOURCE_ROOT'):
  if hasattr(module,attr): assert Path(getattr(module,attr)).resolve()==root,(name,attr)
runtime={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
         if name.startswith('true_rl_agent') and getattr(module,'__file__',None)}
assert all(Path(path).is_relative_to(root/'src') for path in runtime.values())
print(json.dumps(dict(status='PASS',source_root=str(root),module_files=paths,
    runtime_imports_checked=len(runtime),provider_instances_created=0,
    provider_requests=0,gpu_jobs=0,new_n5_results_read=False),indent=2))
