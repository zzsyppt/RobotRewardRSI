"""Offline extraction only; never import or execute a proposal."""
import ast
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWER = ROOT / 'results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer'
OUT = Path(__file__).resolve().parent

# Model semantic judgments. These are category findings, not AR scores.
# Each tuple: category, quote start anchors, mechanism link, testability, ambiguities.
J = {
1: ('present_scope_error', ['Falsifiable expectation on the next public-dev run:'], '倾斜时降低速度阻尼，回到直立后恢复阻尼；预测稳定驻留的中断减少。', 'dwell interruptions 是过程计数，方向明确；但 per pushed episode 与 57 总数混用，使写出的阈值不能直接作为正确归一化的下降标准。', ['57 是 repeated/not_recovered 14 回合、28 计划窗口的中断总数；57/14=4.0714 每该子组回合，57/28=2.0357 每计划窗口。不能把 57 当每 pushed episode 的父基线。', '此项有过程预测，不等于预测的基线表述正确。']),
2: ('present', ['Falsifiable expectation:'], '侧向稳定障碍项针对姿态增长；预测 frontier 的 orientation detected flag 比率下降。', '明确指名 base_orientation_instability_detected，父比率 0.296296，目标 <0.20；区别于 primary base_orientation_instability 终止率。', ['这里的 detected 是过程诊断标记；并未预测其起始时间或持续时长，但冻结规则没有要求每项必须给时间顺序。']),
3: ('present', ['Falsifiable expectation, measured'], '新增 goal_progress_fraction 项，并预测同名进度量在四个具体困难 cell 上升。', 'mean_progress_fraction 是可检验的过程汇总量；四个 cell、方向、父值均明确。具体预测连接了增加目标进度信用的机制。', ['只支持进度增加的预测存在，不证明所谓 tracking 梯度饱和，也不检验其他事实和代码错误。', '聚合进度不能证明先后、停滞持续时间或真实学习根因；但不能因此加门排除过程量。']),
4: ('present', ['Expected falsifiable measurement:'], '修改速度边界外的阻尼形状，预测 planar_speed 违反条件步数下降。', '具名过程计数与机制直接相连；未给数值目标仍可按同套协议比较。', ['观察段把 interruption_condition_counts [168,61,0,0,0] 描述成 episodes carrying at least one violating step；这不是 failed_condition_steps。未来预测的步数必须对应输入中的 [9466,3232,722,401,114]。', '统计窗口受终止截断影响，不能仅据总步数减少推断改善机制；本类别不据此新增失败门。']),
5: ('present', ['Falsifiable expectation for the next training evaluation'], '在接近直立时增加 settling 压力，预测 medium 末端平面速度分布向恢复阈值以下移动。', '变量、方向、cohort 和物理阈值 0.2155 m/s 明确，可比较；不必先给百分位或效应大小才算预测。', ['输入可见两个代表末端速度及计数，并无完整 medium 末端速度分布；完整分布基线不是已核事实。', '“distribution shifted”未指定统计量，属于量化细节不足，不能替作者增补检验。']),
6: ('absent', ['Falsifiable expectation for the next evaluation:'], '改变 pitch/roll 代价，并给成功率、forbidden_contact 和 J_OOD 变化预期。', '明确预测均为终态/分数；便宜 nose-up band 和保持恢复力是奖励函数效果陈述，未承诺轨迹 pitch、clearance、倾斜持续时长或进度如何变化。', ['不能把过程观测引用或机制假说本身自动当未来过程预测。']),
7: ('absent', ['Falsifiable expected measurement:'], '按 posture gate 降低平面阻尼，目标允许 catch-and-settle。', '预测只给各 stratum 成功与 out_of_bounds/fall 数；“keeps full rest regularization”是代码作用，“lets the transient be shaped”是机制意图。', ['未给恢复时长、速度轨迹、dwell 中断等将怎样改变的可检验命题。']),
8: ('ambiguous_intent_only', ['Falsifiable expectation:'], '倾斜时减弱平面阻尼并加强角速度阻尼；模块注释出现 faster rotational arrest。', '正式预期只列结果计数；文档字符串“asks for faster rotational arrest”可读为定性过程方向，也可读为奖励设计目的，未承诺训练后测量。', ['不将其武断计为明确缺失或明确满足；保留全文及模块 docstring，由同一语义口径处理。', '旋转刹停更快若被视为真正行为预期，则无数值时长本身不是否定理由。']),
9: ('ambiguous_runtime_proxy', ['Falsifiable expected measurement:'], '有界姿态门控阻尼；反驳条件提到 recovery_damping 的 runtime 均值停在 no-excess 值 1.0。', '这是一条组件运行值预测/反驳条件，且 reward=1.0 可关联未超 allowance；但未指定训练/评价统计期，也未直接预测速度或持续变化。', ['冻结规则未明确 runtime 组件值是否单独等同机器人过程预测；不能凭有一个 runtime 数字直接解决此边界。', '输入 runtime status=no_prior_runtime_feedback；证据纪律明示 last_mean 只是最后采样快照，不能仅凭它判断组件不活跃或无因果作用。']),
10: ('present', ['Falsifiable expected measurement on the next public-dev run:'], '增加边界外平方 excess 惩罚，预测累计平面速度违反条件步数下降。', '公开全套同协议 baseline 9466 步，方向及样本域明确。', ['9466 是192回合恢复窗口内的非互斥条件步数，观察窗口总步数22169；不是回合数或 repeated 专属值。']),
11: ('absent', ['Expected falsifiable measurement:'], '加陡 near-margin 阻尼斜率；输入观察有后期漂移和恢复后失稳。', '未来测量只有成功数、not_recovered 和 fall 数；残余速度减少属于作者机制目标，但未明确作为将测量的行为变化命题。', ['过程依据较具体不等于已给未来过程预测；不得用观察8/11或速度0.232代替缺失预测。']),
12: ('absent', ['Falsifiable expected measurement:'], '使用姿态门控减少恢复时的速度代价。', '未来只预测 medium 成功率、not_recovered、out_of_bounds/falls；相同于参考/放宽到10%是代码输出条件，不是训练后行为预测。', ['未预测何时回到稳定区、速度怎样衰减或驻留中断怎样变化。']),
13: ('ambiguous_runtime_proxy', ['Falsifiable expectation:'], '脚相对站立脚高度奖励；预期新增项有 non-negligible runtime contribution。', '运行贡献是可观察量，但 non-negligible 未操作化，且非零奖励并不单独表示更高摆脚、更多净空或更短停滞。', ['正式行为结果只预测 frontier 分数；假说正文陈述激励了 lift，但未明确预测实测 lift 的上升。', 'runtime 基线不可用，且输入提醒 last_mean 非完整轨迹或因果证据。']),
14: ('present_qualitative', ['Hypothesis: one bounded reward', 'Falsifiable expectation with one seed-paired run'], '空载脚向上速度奖励明确预测 step-up 时摆腿轨迹抬高、脚和躯干跨边缘净空增加。', '假说段已给具名条件和方向，是可测的轨迹变化；不因其不在 Expectation 标题下排除，也不新加数值阈值门。', ['没有给空间参考、相位窗口、分位数或净空基线；只能认定定性可检验预测存在。', '现有作者可见过程切片不能证明摆腿不足；此审查不将预测存在等同机制对应。']),
15: ('absent', ['Falsifiable expectation:'], '追加 forbidden contact force 成本，推测引导 foot-only support。', '明确未来检验只给成功率/forbidden_contact 终止率和 ID guard；未给 contact force、支撑占比或接触前过程量的变化预期。', ['无禁接触时数值为0是代码性质，不能自动变成未来无禁接触或支撑策略预测。']),
16: ('absent', ['Falsifiable expectation for the next public_dev evaluation:'], '替换为水平/垂向支撑力比超阈值计数，意在避免 stubbing。', '未来只预测 primary forbidden_contact 率与 success；“graded cost before”描述奖励假说，没有预测 stubbing 次数/力比会减少。', ['代码已经计算一个过程量，不意味着作者提出了该过程量训练后变化的预测。']),
17: ('absent', ['Falsifiable expectation under the fixed runner protocol:'], '新增向上脚速度奖励，声明抬脚在奖励上严格更好。', '明确预期只给 stratum success/score、forbidden_contact 和 guards；“makes lifting ... strictly better”为激励排序，无实测抬脚高度/向上速度变化承诺。', ['与014保持同一标准：014明确“raises the swing trajectory ... more clearance”，此项仅称该动作奖励更好。']),
18: ('absent', ['Expected effect:', 'Falsifiable expected measurement:'], '近直立时加大速度惩罚， transient 时按姿态门控。', 'Expected effect 段说梯度约翻倍和动作不会被压制；具名未来测量只有 not_recovered、success、falls。未预测实测晚期速度、稳定持续或中断。', ['“removing ... becomes worthwhile”是奖励价值变化，不等同声明该速度确实将减少。']),
19: ('present_scope_ambiguity', ['Expected measurement (falsifiable):'], '改用 smoothed L1 提升靠近 setpoint 的阻尼，预测 repeated 稳定窗口步比例更高、中断更少。', '具名的过程统计量、方向和 repeated 域明确。', ['父输入分别给 not_recovered 1609/3080、57中断，与 success 1848/2860、8中断；预测未明确是同一结果子组还是整个 repeated。', '不同训练后的 outcome 条件子组成员会变化，不能据子组变化单独推出因果。']),
20: ('present', ['should shorten velocity excursions.', 'Falsifiable expected measurement for the next public_dev run'], '加陡边界外平方项，预测 excursion 更短和 planar 违反条件步数<9466。', '具名过程计数、正确数值可定位、方向明确。', ['9466 对应全体192回合，非 repeated 专属；正文另将 roll/pitch 步数说成0属另一事实核对类别，本报告不宣告硬错误总数。']),
21: ('present', ['Expected measurement:'], '更强速度惩罚预期减少 planar/angular failed-condition steps。', '方向性过程计数预测成立，medium/high/repeated 域可定位；没有阈值不是新增否定门。', ['没有给 pooled 或逐stratum 的明确判据；记录各域父输入计数，不能擅自选择有利聚合。']),
22: ('present_weak_joint_falsifier', ['Expected measurement (falsifiable, this change has not been run):'], 'pitch 成本放宽旨在 slope alignment；反驳句包含 forbidden_contact 与 mean progress 均不变。', 'mean progress 是实际可见过程汇总量，且“均不变则反驳”是含过程量的可检验联合空变化条件；不能直接说完全没有过程预测。', ['仅给联合反驳句，未明确 mean progress 应升高、选 overall 或特定 stratum、以及 pitch/clearance 的轨迹变化；过程承诺明显弱于003。', '本类别认定不代替评价该联结是否足够支持完整 AR2。']),
23: ('absent', ['Falsifiable expectation:'], '改 swing clearance 奖励，声称提供 toward clearance 的直接梯度。', '预期只给成功率/J_OOD/guards；没有实测 clearance 或低前进时长会如何变化的明确行为命题。', ['代码的 clearance proxy 或支持梯度存在不等于实测过程预测。']),
24: ('absent', ['Expected measurement (falsifiable):'], '加入接触力惩罚，期望学会避开 stair face。', '显式未来检验只有 score、completion 和 guards；sustained tilt/inversion 是父观察，没有作为未来过程量预测。', ['“dense graded signal while approaching”是机制陈述，不能自动补出触地前接近过程将变化。']),
25: ('present', ['Hypothesis: edge-concentrated damping', 'Falsifiable expected measurement under the same protocol:'], '稳定边界附近加陡阻尼，预测 repeated/not_recovered 窗口中断率降低、稳定步占比上升。', '过程变量、条件子组、窗口分母和阈值明确：中断/计划窗口<2.0，稳定观察步占比>0.52。', ['父基线57/28=2.035714、1609/3080=0.522403；>0.52 比未四舍五入父值低，故该阈值本身不能确保严格提高。', '结果条件子组会随训练变化，过程变化不独立建立因果。']),
26: ('ambiguous_latch_semantics', ['Falsifiable expectation.'], '增强姿态 barrier，期待减少 forbidden_contact 和 base_orientation_instability latches。', 'forbidden_contact 为终止；base_orientation_instability 不带 detected，按公开字段名是 primary failure。若作者把 latches 指过程检测标记，才有过程事件率，但文本未作区分。', ['不能擅自把 primary 字段改成 base_orientation_instability_detected 来补强预测；保留命名歧义。', '“梯度在0.65之前增加”是代码/机制陈述，不是训练后倾斜过程预测。']),
27: ('present', ['Hypothesis: one coupling cost', 'Falsifiable expectation:'], '倾斜×速度耦合成本预期缩短settling transient、减少 repeated/not_recovered 每回合中断。', '明确过程方向；每not_recovered回合父基线57/14=4.071429，可比较。', ['settling transient时长未给操作化基线；每回合中断已足够独立构成可检验过程量。']),
28: ('absent', ['Expected falsifiable measurement:'], '追加 forbidden-force penalty，期望减少base strike。', '显式未来检验只有 primary forbidden_contact、success、guards；未给力峰值、接触强度/时机或过程量方向。', ['减少终止label不能直接充作减少过程接触次数，尤其一次终止会截断后续。']),
29: ('present_scope_ambiguity', ['Falsifiable expected measurement:'], '增强near-band planar settling，预测晚期速度回升的末端失败模式消失且中断减少。', '具名时间模式与中断方向可检验，机制链接明确。', ['引用medium代表后，预期段切回public_dev；“fewer interruptions”未声明medium/not_recovered或全套统计域。', '输入代表是179步0.114到末端0.232，不能把单代表当完整父分布。']),
30: ('ambiguous_risk_only', ['Expected falsifiable measurement under the same public_dev protocol:', 'Counter-risk made explicit and testable:'], '增加elevation奖励，反风险段称可能提高approach speed并增加forbidden_contact。', '主检验只有成功率/分数；approach speed 是过程量且方向明确，但“could”仅提出可能性，没有出现条件/概率或不出现时反驳路径。', ['不能忽略反风险段，也不能把风险可能性自动当确定的过程预测；未给approach定义、速度基线或窗口。', '父mean progress被引用为观察，但未被写入未来变化预测；不要把它等同003。']),
31: ('absent', ['Falsifiable expected measurement:'], '接触力惩罚被期望在binary termination前给梯度。', '未来预测只有J_OOD、guards和forbidden_contact率；没有接触力轨迹或pre-terminal变化预测。', ['before termination 的奖励作用断言不是过程先后预测；此类别不验证该断言正确性。']),
32: ('absent', ['Falsifiable expectation:'], '增加stance-relative swing lift奖励以改善clearance。', '未来只给forbidden_contact/J_OOD/guards；“should help clear taller risers”是一般动作目的，没有具名高度/持续/过程统计如何变化的命题。', ['引用failed-cell mean progress仅是观察；没有mean progress上升预测。']),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_ra(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'rationale':
            return node.value, ast.literal_eval(node.value)
    raise ValueError('rationale absent')

def evidence(value, path):
    return {'json_path': path, 'value': value}

def baseline(s):
    f=s['public_feedback_summary']; result=[]
    if 'public_dev' in f:
        d=f['public_dev']; base='$.public_feedback_summary.public_dev'
        for k in ('score_stats','recovery_counts','stratum_results','failure_breakdown'):
            result.append(evidence(d.get(k),base+'.'+k))
        b=d.get('behavior',{}); bp=base+'.behavior'
        for k in ('scope','measurement_contract','not_recovered_vs_success','definitions'):
            if k in b: result.append(evidence(b[k],bp+'.'+k))
        if 'profile' in b:
            p=b['profile']; rows=[]
            for idx,row in enumerate(p.get('rows',[])):
                if row[0] in ('all','medium','high','repeated'):
                    rows.append({'row_index':idx,'named_values':dict(zip(p['columns'],row))})
            result.append(evidence({'condition_order':p.get('condition_order'),'scope':p.get('scope'),'rows':rows},bp+'.profile'))
        for idx,v in enumerate(b.get('representatives',[])):
            result.append(evidence({k:v[k] for k in ('episode','outcome','terminal_step','terminal','final_failed_conditions') if k in v},bp+f'.representatives[{idx}]'))
        if not b or b.get('status')=='not_available': result.append(evidence(b,bp))
    else:
        base='$.public_feedback_summary.feedback_entries[0].feedback'
        d=f['feedback_entries'][0]['feedback']; g=d['generalization_summary']
        result.append(evidence(g['episodes_per_cell'],base+'.generalization_summary.episodes_per_cell'))
        result.append(evidence(g['success_count_by_cell'],base+'.generalization_summary.success_count_by_cell'))
        result.append(evidence(g['stratum_diagnostics'],base+'.generalization_summary.stratum_diagnostics'))
        result.append(evidence(d['progress_summary'],base+'.progress_summary'))
        result.append(evidence(d['diagnostic_flag_rates'],base+'.diagnostic_flag_rates'))
        result.append(evidence(f['diagnostic_table_definitions']['stratum_diagnostics'],'$.public_feedback_summary.diagnostic_table_definitions.stratum_diagnostics'))
        b=f.get('behavior',{}); bp='$.public_feedback_summary.behavior'
        for k in ('scope','definitions','by_outcome'):
            if k in b: result.append(evidence(b[k],bp+'.'+k))
        if 'profile' in b:
            for k in ('profile','semantics'):
                if k in b['profile']: result.append(evidence(b['profile'][k],bp+'.profile.'+k))
        if not b: result.append(evidence(None,bp+' (absent)'))
    result.append(evidence(s.get('runtime_feedback_summary'),'$.runtime_feedback_summary'))
    return result

records=[]
for i in range(1,33):
    stem=f'proposal-{i:03}'; p=REVIEWER/(stem+'.py'); src=p.read_text(encoding='utf-8'); tree=ast.parse(src)
    parent=REVIEWER/(stem+'-parent.py'); parent_src=parent.read_text(encoding='utf-8'); pt=ast.parse(parent_src)
    seen_p=REVIEWER/(stem+'-seen.json'); init_p=REVIEWER/(stem+'-initial.json')
    seen=json.loads(seen_p.read_text(encoding='utf-8')); initial=json.loads(init_p.read_text(encoding='utf-8'))
    node,rationale=get_ra(tree); status,anchors,mechanism,testability,ambiguities=J[i]
    quotes=[]
    for anchor in anchors:
        # Keep from exact anchor through paragraph end. Full rationale is retained as well.
        start=rationale.index(anchor); end=rationale.find('\n',start)
        if end < 0: end=len(rationale)
        quotes.append({'quote':rationale[start:end], 'location':'rationale', 'source_start_line':node.lineno,'source_end_line':node.end_lineno})
    doc=ast.get_docstring(tree,clean=False)
    if i==8: quotes.append({'quote':doc,'location':'module_docstring','source_start_line':tree.body[0].lineno,'source_end_line':tree.body[0].end_lineno})
    def norm(x):
        return ast.dump(ast.Module(body=x.body,type_ignores=[]),include_attributes=False)
    pfunc={x.name:x for x in pt.body if isinstance(x,ast.FunctionDef)}
    changes=[]
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name!='build_training_components' and (fn.name not in pfunc or norm(fn)!=norm(pfunc[fn.name])):
            changes.append({'function':fn.name,'start_line':fn.lineno,'end_line':fn.end_lineno,'source':ast.get_source_segment(src,fn)})
    build=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_training_components')
    term_source=[]
    for n in ast.walk(build):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='RewardTerm':
            term_source.append({'line':n.lineno,'source':ast.get_source_segment(src,n)})
    records.append({
        'proposal_id':stem,
        'category_judgment':status,
        'only_process_prediction_category':True,
        'full_AR_or_mechanism_correspondence_assigned':False,
        'exact_prediction_or_nearest_claim_quotes':quotes,
        'full_rationale_exact_python_string':rationale,
        'rationale_raw_source':ast.get_source_segment(src,node),
        'rationale_source_lines':[node.lineno,node.end_lineno],
        'mechanism_connection':mechanism,
        'testability_judgment':testability,
        'ambiguities_and_limits':ambiguities,
        'visible_baselines_and_denominators':baseline(seen),
        'baseline_evidence_role':'These values are present in this proposal actual seen input. They are not newly recomputed experimental truth, not inferred author visibility from another file.',
        'changed_callable_sources':changes,
        'reward_term_sources':term_source,
        'input_provenance':{
            'initial_equals_seen':initial==seen,
            'changed_top_level_keys':sorted(k for k in set(initial)|set(seen) if initial.get(k)!=seen.get(k)),
            'files':[{'path':str(x.relative_to(ROOT)), 'sha256':sha(x)} for x in (p,parent,seen_p,init_p)]
        },
    })

counts=dict(collections.Counter(r['category_judgment'] for r in records))
obj={
    'schema_version':'p3r.posthoc_process_prediction_category_review.v1',
    'created_date_local':'2026-09-29',
    'status':'posthoc_category_audit_not_independent_new_experiment',
    'reviewer_identity':'Codex subagent /root/r0_process_prediction_audit; backend version not independently attested',
    'purpose':'Apply the frozen process-prediction clause consistently to all 32 anonymous proposals without assigning complete AR scores or any gate.',
    'rules_source':{'path':'R0B_REVIEW_RULES_ZH.md','sha256':sha(ROOT/'R0B_REVIEW_RULES_ZH.md'), 'frozen_clause':'并给出可检验的过程预测'},
    'method':[
        'Read only frozen rules and the 32 proposal, parent, seen and initial files in the designated reviewer bundle. No withhold mapping, old review scores/aggregate, Audit/rescore documents, API calls, simulation or proposal execution.',
        'All 128 allowed evidence files parsed; all 32 rationales semantically reviewed; AST source comparison is used only to locate the mechanism and preserve exact quotes, not to claim complete hard-error auditing.',
        'A process prediction concerns a measurable state, rate, duration, order, persistence, or conditioned trajectory/statistic. Outcome success, score and primary termination rates alone do not suffice.',
        'Search the whole rationale and module description; the statement need not occur in a section labeled expectation. Separate past observation, analytic reward effect, general prior, proposed causal hypothesis, and future predicted measurement.',
        'No new requirement for numeric effect size, p-value, a temporal sequence in every proposal, or process terms outside aggregated progress. Directional measurable changes can qualify.',
        'mean_progress_fraction can count. Specific named cells and direction in 003 are testable and mechanism-linked; a weak joint null condition in 022 is retained with its limitations; merely citing mean progress in 030 or 032 does not manufacture a future progress prediction.',
        'Runtime reward summaries, intent-only language, ambiguous latches and possibility-only counter-risks retain explicit ambiguous categories; these are not quietly converted into full AR decisions.',
        'Record all numerator/denominator mismatches seen within this category and preserve missing baselines as unavailable. Do not replace aggregate counts by per-episode or per-window units.',
        'Category presence does not establish correct process evidence, mechanism compatibility, authorization, actual efficacy, absence of other hard errors, or AR2.'
    ],
    'category_counts_not_AR_counts':counts,
    'proposal_count':len(records),
    'original_scores_or_evidence_modified':False,
    'independent_training_or_new_frozen_experiment':False,
    'records':records,
}
dest=OUT/'process_prediction_all32.json'; dest.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# R0-b 全体过程预测类别审查方法（2026-09-29）','','这是**事后类别审查，不是独立新实验**。只审冻结 AR2 中“并给出可检验的过程预测”这一必要部分；不赋总 AR、机制对应或门结论，不改原分数。','','材料限定为冻结规则与指定 reviewer 包的 32 份提案、32 份父源码、32 份 seen、32 份 initial。没有读取 withheld mapping、旧评阅分数/aggregate、Audit/rescore 文档；没有调用模型 API、仿真或执行提案。128 文件均作结构读取，32 个 rationale 均作语义复核。初次/最终输入只有 010、030 不同。源码经 AST 只读解析，记录原文、行号、SHA256、机制 callable 与 RewardTerm；不声称完成全部事实/代码硬错误核对。','','统一口径：过程量可以是速度、姿态、进度、持续/先后、条件违反步数、驻留中断或条件下的轨迹变化。成功率、综合分数、primary 终止率单独只是结果预测；已有过程观察、奖励公式性质和泛化先验也不能自动转为未来过程预测。不限于 Expectation 标题，不额外要求数值效应阈值、显著性检验或每项必须写时间顺序。','','mean progress 允许作为过程量：003 给出四个具体 cell、父值与上升方向，直接连到 progress 奖励；022 在联合反驳句中提到 mean progress 不变，保留为较弱的过程检验及范围歧义。030/032 只在观察中给进度，不能补造进度预测。014 在 Hypothesis 段明确预期摆腿轨迹抬高与净空增加，应记录为定性过程预测。','','运行项贡献、设计意图、latch 命名与 could 风险句的边界统一保留 ambiguous 类别，不为得到整齐二分而投票。009/013 为 runtime proxy；008 为 asks 的意图/行为歧义；026 未区分 primary termination 与 detected；030 为风险可能性。过程预测存在与预测基线正确、过程支持正确、代码兑现和全部 AR2 是不同问题。','','特别可复核项：001 将57总中断数写成 per pushed episode 阈值；实际为 repeated/not_recovered 的14回合、28计划窗口（57/14=4.0714，57/28=2.0357）。025 的稳定占比 >0.52 低于真实父值1609/3080=0.522403，因此阈值本身不保证严格提高。019/029 的子组/聚合范围也需保留歧义。完整数值和JSON路径见同目录 process_prediction_all32.json。','','下面仅列过程预测类别，不能相加推导任何变体有效、AR2数量或固定出口。','','| 提案 | 类别 |','|---|---|']
lines += [f"| {r['proposal_id']} | {r['category_judgment']} |" for r in records]
lines += ['','类别中 present 表示找到可检验过程命题；附加后缀说明范围/量化限制。absent 表示只找到结果预期、代码作用或一般意图。ambiguous 表示原句本身存在指定语义边界，不能靠增加新规则消除。所有判断保留提案完整 rationale 与精确摘录，供复核。','']
(OUT/'process_prediction_method_zh.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'written':str(dest),'records':len(records),'counts':counts,'bytes':dest.stat().st_size},ensure_ascii=False))
