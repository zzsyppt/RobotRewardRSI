"""Post-hoc category audit, never an R0-b rescore or a new confirmation run.

Reads exactly the 32 final reviewer sources, parents, initial/seen JSON files,
and the preregistered review rules. Executes pure candidate callables on small
NumPy CPU arrays after omitting imports; no production module, API or simulator
is loaded. Formulas are checked against the original callable, not a rewrite.
Run: .venv/Scripts/python.exe PLAN/feedback_sys/P3R/audit_response_20260929/recompute_preservation_claims.py
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer"
OUT = Path(__file__).with_name("preservation_claim_all32.json")


class Tensor(np.ndarray):
    """Only arithmetic used by the selected pure-callable counterexamples."""
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def square(self): return Tensor(np.square(np.asarray(self)))
    def sqrt(self): return Tensor(np.sqrt(np.asarray(self)))
    def exp(self): return Tensor(np.exp(np.asarray(self)))
    def abs(self): return Tensor(np.abs(np.asarray(self)))
    def float(self): return Tensor(np.asarray(self, dtype=float))
    def clamp(self, min=None, max=None):
        return Tensor(np.clip(np.asarray(self), -np.inf if min is None else min, np.inf if max is None else max))
    def sum(self, dim=None, keepdim=False): return Tensor(np.asarray(self).sum(axis=dim, keepdims=keepdim))
    def mean(self, dim=None): return Tensor(np.asarray(self).mean(axis=dim))
    def norm(self, dim=None): return Tensor(np.linalg.norm(np.asarray(self), axis=dim))
    def unsqueeze(self, dim): return Tensor(np.expand_dims(np.asarray(self), dim))
    def max(self, dim=None, keepdim=False):
        return NS(values=Tensor(np.asarray(self).max(axis=dim, keepdims=keepdim)))
    def min(self, dim=None, keepdim=False):
        return NS(values=Tensor(np.asarray(self).min(axis=dim, keepdims=keepdim)))
    def amax(self, dim=None): return Tensor(np.asarray(self).max(axis=dim))


def constructor(**kw): return NS(**kw)


def load_source(text, filename):
    tree = ast.parse(text, filename)
    safe = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    scope = {"RewardInputs": object, "RewardTerm": constructor, "TrainingComponents": constructor}
    exec(compile(safe, filename, "exec"), scope)
    return tree, scope, scope["build_training_components"]()


def inp(roll=0.0, pitch=0.0, height=1.0, planar=0.0, angular=0.0, **other):
    d = dict(roll=Tensor([roll]), pitch=Tensor([pitch]), base_height_relative=Tensor([height]),
             base_lin_vel_scaled=Tensor([[planar, 0.0, 0.0]]),
             base_ang_vel_scaled=Tensor([[angular, 0.0, 0.0]]),
             joint_pos_error=Tensor([[0.0]*12]), action_delta=Tensor([[0.0]*12]),
             terminated_non_timeout=Tensor([False]),
             projected_gravity=Tensor([[math.sin(pitch), -math.sin(roll)*math.cos(pitch), -math.cos(roll)*math.cos(pitch)]]))
    d.update({k:Tensor(v) for k,v in other.items()})
    return NS(**d)


def scalar(x): return float(np.asarray(x).reshape(-1)[0])


def fn_body(tree, name):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = copy.deepcopy(node.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    return ast.dump(ast.Module(body=body,type_ignores=[]), include_attributes=False)


records = {}
inventory = []
for i in range(1,33):
    key = f"proposal-{i:03d}"
    content = {}
    for suffix in (".py", "-parent.py", "-initial.json", "-seen.json"):
        p = REVIEWER / (key + suffix)
        raw = p.read_bytes()
        content[suffix] = raw.decode("utf-8-sig")
        inventory.append({"file":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
    tree,scope,program = load_source(content[".py"],key+".py")
    ptree,pscope,pprogram = load_source(content["-parent.py"],key+"-parent.py")
    initial = json.loads(content["-initial.json"])
    seen = json.loads(content["-seen.json"])
    common = []
    old = {t.name:t for t in pprogram.reward_terms}
    new = {t.name:t for t in program.reward_terms}
    for name in sorted(old.keys() & new.keys()):
        a,b=old[name],new[name]
        common.append({"term":name,"function_body_equal_ignoring_docstrings_and_annotations":fn_body(ptree,a.fn.__name__)==fn_body(tree,b.fn.__name__),
                       "weight_equal":a.weight==b.weight,"clips_equal":all(getattr(a,k,None)==getattr(b,k,None) for k in ("clip_min","clip_max"))})
    records[i]={"id":key,"tree":tree,"scope":scope,"program":program,"ptree":ptree,"pscope":pscope,"pprogram":pprogram,
                "initial":initial,"seen":seen,"source":content[".py"],
                "result":{"proposal_id":key,"category":"preservation_or_activation_scope_or_cost_reduction_claim",
                          "initial_and_seen_json_equal":initial==seen,"parent_comparison":{
                              "common_terms":common,"removed_term_names":sorted(old.keys()-new.keys()),"added_term_names":sorted(new.keys()-old.keys()),
                              "comparison_note":"Semantic function bodies, weights and clips. Added type annotations do not constitute changed reward mechanisms; literal byte-identical wording is not separately rescored."},
                          "claims":[]}}
    records[i]["result"]["callable_source_locations"]=[
        {"name":n.name,"file":f"results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer/{key}.py",
         "start_line":n.lineno,"end_line":n.end_lineno}
        for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!="build_training_components"]
    records[i]["result"]["reward_term_configuration"]=[
        {"name":t.name,"weight":t.weight,"function":t.fn.__name__,
         "clip_min":getattr(t,"clip_min",None),"clip_max":getattr(t,"clip_max",None)} for t in program.reward_terms]


def quote(i, phrase):
    r=records[i]
    for n in ast.walk(r["tree"]):
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            match=re.search(re.escape(phrase).replace(r"\ ",r"\s+"),n.value)
            if match:
                return {"file":f"results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer/proposal-{i:03d}.py",
                        "literal_start_line":n.lineno,"literal_end_line":n.end_lineno,"quote":match.group(0)}
    raise AssertionError((i,phrase))


def claim(i, phrase, status, scope, analysis, checks=None, ambiguity=None):
    c={"evidence":quote(i,phrase),"status":status,"claim_scope":scope,"analysis_zh":analysis}
    if checks: c["cpu_counterexample_or_check"]=checks
    if ambiguity: c["ambiguity_or_limit_zh"]=ambiguity
    records[i]["result"]["claims"].append(c)


def call(i, fn, x, parent=False): return scalar(records[i]["pscope" if parent else "scope"][fn](x))


def weighted(i, fn, x):
    t=next(t for t in records[i]["program"].reward_terms if t.fn.__name__==fn)
    v=call(i,fn,x)
    return t.weight*max(getattr(t,"clip_min",-math.inf),min(getattr(t,"clip_max",math.inf),v))


stable_x=inp(roll=.1,planar=.04)
stable_limits=records[1]["seen"]["public_feedback_summary"]["public_dev"]["behavior"]["measurement_contract"]["stable_limits"]
scale=records[1]["seen"]["task"]["spatial_scale"]["velocity_scale_mps"]
stable_check={"roll_rad":.1,"pitch_rad":0,"height_relative":1,"height_m":.74,"planar_scaled":.04,
              "planar_mps":.04*scale,"angular_speed":0,"inside_published_stable_limits":True,
              "published_limits":stable_limits,"dwell_note":"A state inside the stable set; repeating it for the declared 21 steps stays inside the dwell band. This is algebra, not a simulated trajectory."}
assert .1 < stable_limits["recovery_tilt_rad"] and .04*scale < stable_limits["recovery_velocity_mps"] and .74>stable_limits["recovery_height_m"]

p1=call(1,"stance_damping",stable_x); p1old=call(1,"stationary_v3",stable_x,True)
assert p1!=p1old
claim(1,"the upright regime keeps full dwell-aligned damping because that is the regime the task judges for dwell.",
      "scope_ambiguity_not_hard_error","undefined upright regime versus entire stable/dwell band",
      "若 upright regime 指整个任务稳定集合，则此句不成立：稳定集合内 roll=.1 时 keep=.8125，速度成本减弱18.75%。若仅指 roll=pitch=0，则姿态门确实为1；文字没有给 upright regime 精确定义，不能把整个 dwell 的读法当作无歧义硬错。",
      {**stable_check,"keep":.8125,"parent_reward_rate":p1old,"candidate_reward_rate":p1},
      "零姿态点与整个稳定区间分开。另有明示 excess cap，因此全速度域的零姿态点也不保证与未裁剪父项完全一致；degrades gracefully 不按 exact identity 判。")
claim(2,"Every other reward function, weight and clip is unchanged from the reference.","supported_in_scope","all reward mechanisms except the replaced orientation term","七个共同项的函数体、权重和clip均一致；替换姿态项被明确排除。")
claim(3,"all eight parent callables, their weights and their absence of clips are preserved exactly","supported_in_scope","the eight existing RewardTerms","新增progress之外八个父项均保持。假说中的 exactly where 未声明硬门；代码全域新增progress，不将训练解释当已证。")
claim(4,"in-band behavior is unchanged.","supported_in_scope","both normalized velocity channels <= 1","两通道excess均为0时新增excess平方为0，外部[-5,1] clip在in-band输出[.75,1]不生效；原项保持。高速度尾部另有显式cap，不将其误扩大为全域不变。",
      {"parent":call(4,"stationary_v3",stable_x,True),"candidate":call(4,"band_damping",stable_x)})

x5=inp(roll=.4,planar=.08)
v5=weighted(5,"settle_hold",x5)
assert v5<0
claim(5,"a posture-gated settling rate that is inactive while the trunk is tilted or the base height is far from nominal.",
      "algebraic_contradiction","added settle_hold term is inactive under tilt or far-from-nominal height",
      "exp(-2.5*tilt_sq-3*height_error_sq)在所有有限误差下严格正；只要速度excess>0就仍收费，clip也不会把小负数归零。是衰减而非关闭。",
      {"input":{"roll":.4,"pitch":0,"height_relative":1,"planar_scaled":.08,"angular_scaled":0},"formula":"-exp(-2.5*0.4^2)*(1-0.4)^2/((1-0.4)+0.4)","added_weighted_reward_rate":v5},
      "此处 inactive 是精确零声明；是否能产生训练中的 stiffness 是未训练假说，不由反例判断。")
claim(6,"keeps the same lateral-tilt pressure","scope_ambiguity_not_hard_error","lateral cost versus coefficient versus approximate small-angle pressure",
      "roll项从sin(roll)^2*cos(pitch)^2变为roll^2，因此整个横向函数并不相同。但same pressure可能指同一权重或近零二阶强度，且上下文明确说steepening；不强制解释为全域恒等。",
      {"roll":.4,"pitch":0,"old_lateral_cost":math.sin(.4)**2,"new_lateral_cost":.4**2},
      "不在此类别重新审查raw bound、角度公式数值或每步dt陈述。")

p7=call(7,"stationary_v3",stable_x); p7old=call(7,"stationary_v3",stable_x,True)
assert p7>p7old
claim(7,"keeps full rest regularization exactly where the dwell is measured","algebraic_contradiction","full planar rest regularization in the task dwell region",
      "该句明确把full保持范围绑定到dwell区域；但g=.25+.75*exp(-6*(roll^2+pitch^2+(hrel-1)^2))仅在姿态误差和高度误差同时精确为0时等于1。非零小误差仍在stable band内，却已经减弱平移成本。角阻尼确实不变。",
      {**stable_check,"gate":.25+.75*math.exp(-6*.1**2),"parent_reward_rate":p7old,"candidate_reward_rate":p7},
      "这是完整dwell区域声明，与012只在nominal零点相同及strong表述不同；不推出训练收益或危害。")

x8level=inp(planar=2); x8tilt=inp(roll=.4,planar=2)
total8=lambda x:3*call(8,"upright_height",x)+call(8,"tilt_conditioned_damping",x)
a8,b8=total8(x8level),total8(x8tilt)
assert b8>a8
claim(8,"At tilt of zero the new term is numerically identical to the reference term","supported_in_scope","roll=pitch=0 exact point for any speeds","此时planar_gate=angular_gain=1，逐式等于父项。不能把这句扩大成整个dwell区间保持。")
claim(8,"The posture term upright_height keeps weight 3.0 and still dominates the damping swing, so a tilted and moving base remains strictly worse than a level one and no falling incentive is created.",
      "algebraic_contradiction","global reward ordering for otherwise identical level versus tilted moving states",
      "姿态项损失有界于3，平移Huber成本无上界；倾斜减小平移成本所释放的奖励可以超过姿态损失。同速同高度同零角速度比较即可反向。",
      {"level_input":{"roll":0,"planar_scaled":2,"angular_scaled":0,"height_relative":1},"tilted_input":{"roll":.4,"planar_scaled":2,"angular_scaled":0,"height_relative":1},
       "level_upright_plus_damping_rate":a8,"tilted_upright_plus_damping_rate":b8,"tilted_minus_level":b8-a8,
       "remaining_terms":"identically zero: joint error, action delta, terminated_non_timeout"},
      "仅反驳无条件代数支配保证。输入scaled速度2约5.39m/s未宣称在已记录轨迹中出现；不判断训练后策略会倾斜或跌倒。")
claim(9,"keeps standing damped (allowance stays 0.10 when upright at nominal height)","supported_in_scope","a positive planar allowance at exact nominal posture; not equivalence to parent","nominal时allowance=.10；超过allowance有有界成本。并没有声称整个stable band数值等同父项。caps multi-step price仅逐步有界、对固定步数有界，不视为任意时长总成本固定。")
claim(10,"Inside the bounds the hinge is zero with zero gradient, so sub-threshold motion is not penalized.","supported_in_scope","the newly added hinge only","两速度ratio<=1时新增项及其一阶斜率为0。原stationary_v3仍在阈内收费，但上下文已明确主语是hinge，不把此句误读为全部程序不收费。",
      {"added_hinge":call(10,"bound_excess",stable_x)})

x11=inp(planar=.24); old11=call(11,"stationary_v3",x11,True); new11=call(11,"stationary_margin_brake",x11)
assert new11<old11
claim(11,"lowers the cost of high-speed transient motion that may be needed to arrest momentum.","algebraic_contradiction","total planar cost in the stated >2x-margin high-speed branch",
      "r>2时父成本r-.5，新成本6.5+.25*(r-2)=6+.25r；新-父=6.5-.75r。故2<r<26/3时成本更高，只有r>26/3时总成本才更低。低尾部斜率(.25<1)是真，但不能推出整个尾段总成本下降。",
      {"ratio":3,"planar_scaled":.24,"parent_planar_cost":2.5,"candidate_planar_cost":6.75,"parent_reward_rate":old11,"candidate_reward_rate":new11,"cost_crossing_ratio":26/3},
      "若仅主张marginal cost下降则成立；被反驳的是docstring明确的lowers the cost，非数值重评分。")
claim(12,"When the robot is upright at nominal height the gate is 1.0 and the term is identical to the reference.","supported_in_scope","exact nominal point: roll=pitch=0 and height_relative=1","该精确零点上gate=1成立。稳定区间内其它点gate<1不反驳零点声明。后文strong stationary damping没有数值定义，也没有full/unchanged区域声明，不据此硬判。",
      {"nominal_parent":call(12,"stationary_v3",inp(planar=.04),True),"nominal_candidate":call(12,"stationary_v3",inp(planar=.04)),"in_band_nonzero_roll_gate":.96},
      "strong是定性；是否足够强属于待训练假说。")

x13=inp(support_pos_relative_to_terrain_origin=[[[0,0,0],[0,0,.14],[0,0,.14],[0,0,.14]]],support_contact=[[1,0,0,0]])
raw13=call(13,"swing_step_up_clearance",x13); w13=weighted(13,"swing_step_up_clearance",x13)
assert raw13>2 and w13==1
claim(13,"is attained only when two support effectors are simultaneously at least 0.2 m above the highest contacting effector.",
      "algebraic_contradiction","necessary condition for the clipped maximum contribution",
      "代码跨所有未接触脚求和后clip到2；一脚接触、其余三脚各高.14m时raw=2.1，已到最大weighted=1，但没有任何脚达到.2m。两脚各.2是充分例子，不是必要条件。",
      {"contact":[1,0,0,0],"heights_m":[0,.14,.14,.14],"raw":raw13,"clipped":2,"weighted_rate":w13,"effectors_at_least_point2_above_contact":0})
claim(14,"the eight reference callables, their weights and their absence of clips are unchanged","supported_in_scope","eight parent terms and per-effector swing gate","八个父项不变；新增项air_time>0且vz>0且水平command norm>.1才收费。stance-phase untouched可仅指加载脚贡献为0；并不证明整条ID步态/未来策略不变。")
claim(15,"no currently successful trajectory return changes","unsupported_historical_preservation_guarantee","all actually successful historical trajectories",
      "零forbidden force时新增项确为0；但任务失败谓词有正force阈值，未失败不推出force恰好0。F=.5N即产生-1奖励率且可以低于1N latch阈值。这否定从标签推导恒零的逻辑保证；没有逐步历史成功force档案，不能宣称某条实际历史轨迹已被改价。",
      {"force_norm_n":.5,"added_weighted_rate":weighted(15,"forbidden_contact_force_l2",inp(forbidden_contact_force=[[[.5,0,0]]])),"threshold_n":1.0},
      "不计为已复核的历史事实硬错。实际成功轨迹是否有非零force仍未核。")
claim(16,"the term is numerically inert on the behaviour that already solves C0-ID and C1-ID","unsupported_historical_preservation_guarantee","actual ID behavior, not all mathematically possible clean-contact arrays","当前窗口内每脚horizontal<=4*abs(vertical)时新stumble成本确为0；成功标签本身不包含该比值恒不越界的证明。历史力比值未逐步核验，不能直接硬判实际轨迹非零。其余七个保留父项已结构核验。")

x17=inp(support_air_time=[[.1,0,0,0]],support_lin_vel=[[[0,0,.5],[0,0,0],[0,0,0],[0,0,0]]],commanded_base_vel=[[0,.2,0]])
v17=call(17,"swing_foot_lift",x17)
assert v17>0
claim(17,"gated by commanded forward speed above 0.1 m/s.","scope_ambiguity_not_hard_error","forward x component versus horizontal command magnitude","代码使用norm(commanded_base_vel[:,:2])，并非只用+x。若forward精确指+x则纯侧向command也激活，是反例；但作者可能把forward speed宽泛指水平移动速度，且是否实际协议只给+x未知，保留术语歧义而不扩成硬错。",
      {"commanded_base_vel":[0,.2,0],"forward_x":0,"horizontal_norm":.2,"raw_added":v17})
claim(17,"the term is exactly zero while a foot is loaded, so the flat-stair gait that earns c0_id and c1_id a perfect score is not reweighted.","unsupported_historical_preservation_guarantee","per-foot loaded state versus whole successful gait","每脚air_time=0时该脚贡献为0；全体求和仍可由其它swing脚贡献。不能由loaded脚为零推出包含swing相位的整条ID轨迹不变；具体历史ID swing数据未核，不把预期守住ID当已观察收益。")
claim(18,"while during a large-excursion transient the gate is about 0.5","scope_ambiguity_not_hard_error","undefined large-excursion region and approximate gate value","g=exp(-6E)仅E=ln2/6时=.5；large-excursion没有指定E范围且此句处于Expected effect，不把任意较大E计算成无歧义硬错。近零处加倍斜率成立，其余四项保持。")
claim(19,"The far-field slope stays one, so pressure to arrest large transients is preserved","supported_asymptotic_not_exact_finite_identity","far-field asymptotic derivative","新成本导数r/sqrt(r^2+.0025)趋于1，父Huber尾导数为1；按far-field的渐近语义成立。任意有限r存在小差异不等于该句硬错；没有全域成本相等声明。")
claim(20,"keeps the identical in-bound quadratic 0.5 * clamp(x, max=1.0)^2 and the identical linear excess (x - clamp(x, max=1.0))","supported_in_scope","two existing cost components, with an explicitly added excess-square component","新增平方excess在in-bound为0，原两部分逐式保留。其它数字与过程引用不在本次类别重评。")
claim(21,"while removing the positive stationary baseline.","supported_in_scope","removal of stationary +1 baseline and preservation of other terms","代码从1-.25*C改为-3*C，+1确实删除。其余四项保持。3x倍率属于另一个数值口径，不在本类别重新评分。")
claim(22,"keeps the roll component at the anchor's full strength","supported_in_scope","projected_gravity[:,1]^2 component, not Euler roll independent of pitch","此项仍为g_y^2、权重-2.5，无改动；cos(pitch)几何耦合也仍在。不能把full strength自行解释为去除原有pitch耦合。")
claim(23,"while leaving flat-ground locomotion nearly untouched, because on flat ground the lift stays small.","untrained_or_approximate_hypothesis_not_hard_error","approximate future flat-ground locomotion preservation","替换air_time为clearance确实会改变gait奖励；nearly untouched没有误差范围且处于Hypothesis，不当作精确恒等或已证收益。明确声明保持的其余七项结构上成立。")
claim(24,"exactly zero when that force is zero.","supported_in_scope","forbidden force equals exact zero","1-exp(-0)=0，negative weight与clip不改变该零点。并未像015/028把零成本扩大为所有成功/无失败区域。")

x25=inp(roll=.1,angular=.25); x25z=inp(angular=.25)
diff25=call(25,"recovery_damping",x25)-call(25,"recovery_damping",x25z)
assert diff25<0
claim(25,"charges angular rate only while roll and pitch exceed the 0.2 rad limit.","algebraic_contradiction","activation restricted to tilt-threshold violations","tilt=clamp((roll^2+pitch^2)/.04,max=1)没有减阈值或正部hinge。任意非零姿态误差、非零angular_core都会产生额外成本；roll=.1、pitch=0均未越限而已收费。",
      {"roll_rad":.1,"pitch_rad":0,"angular_scaled":.25,"angular_core":.5,"tilt_gate":.25,"extra_weighted_reward_rate":diff25,"formula":"-.25*(.1^2/.04)*.5 = -.03125"})
claim(25,"keeps the outside-band tail linear at 0.75 per limit unit so the unavoidable post-impact transient is not made more dominant","scope_ambiguity_not_hard_error","tail slope versus total out-of-band cost and qualitative dominance","斜率.75成立，但本项在r>1的基础成本2+.75*(r-1)比父r-.5高至r=7才交叉。dominant缺少明确总奖励/速度域定义，不由斜率直接证明，也不在此扩成无歧义硬错。",
      {"ratio":3,"old_cost":2.5,"new_base_cost_without_tilt":3.5,"base_cost_crossing_ratio":7})
claim(26,"it keeps the identical small-angle L2 regulation below a deadband at tilt_sq = 0.12","scope_ambiguity_not_hard_error","small-angle second-order approximation versus exact equality throughout deadband","父项sin(p)^2+sin(r)^2*cos(p)^2与r^2+p^2不是有限区间恒等，但二阶small-angle展开一致。identical与small-angle并列有歧义；未定义精确误差容差，不能自动把二阶近似说法硬判为全区间恒等错误。",
      {"roll":.3,"pitch":0,"tilt_sq":.09,"deadband":.12,"parent":call(26,"flat_orientation_l2",inp(roll=.3),True),"candidate":call(26,"stair_attitude_barrier",inp(roll=.3))})
claim(27,"keep upright_height, stationary_v3, joint_pose, action_rate and fall byte-identical to the anchor","supported_in_scope","all five original reward mechanisms","五个父项函数体、权重、clip均保持；新增耦合在精确tilt=0或两速度=0时为0，并未声称整个stable band关闭。")
claim(28,"it is inactive where that failure kind never occurs.","unsupported_historical_preservation_guarantee","strata with zero forbidden_contact primary-failure count","失败标签为0不保证当前resolved force处处为0：F=.5N<1N时新增cost仍为-.1。该反例否定从失败率到失活的必然推导；未逐步核验实际零失败strata的力，不裁定其历史具体轨迹已被改价。",
      {"force_norm_n":.5,"threshold_n":1.0,"added_weighted_rate":weighted(28,"forbidden_contact_force_penalty",inp(forbidden_contact_force=[[[.5,0,0]]]))},
      "其余八项保留成立；只有force=0保证新增项零。")
claim(29,"the angular shape is kept identical.","supported_in_scope","angular velocity damping component","angular_demand=父Huber成本，外系数.25及weight=1保持。另一句bounded pressure可指有界marginal slope，不自行解释为线性尾总成本有界。")
claim(30,"the added term is a near-constant offset there and introduces almost no new directional incentive on flat support.","untrained_or_approximate_hypothesis_not_hard_error","approximate flat-support behavior, not exact constant reward","在base_height固定的局部比较上为常数；在真实平地不同姿态/高度间不恒定。near/almost与nominal standing限定是经验近似，不构成精确恒等保证；是否训练改变姿态需新实验。八个父项保持。")
claim(30,"so the added term cannot dominate the reward.","scope_ambiguity_not_hard_error","undefined dominance: ceiling comparison versus actual contribution or training influence",
      "2<3.5仅比较两个项的上界，并不能保证实际时刻elevation小于tracking。比如height=1.5、线/角跟踪平方误差均为1时，新增率1.5，tracking合计4.25*exp(-4)=.07784。若dominate定义为实际贡献最大，该保证不成立；若指训练影响，没有静态定义或实验证据，保留歧义，不硬定因果。",
      {"base_height":1.5,"new_rate":1.5,"linear_and_angular_tracking_rates":4.25*math.exp(-4)},
      "与008明确的strictly worse全局排序不同，此处dominate没有操作化定义，不能把贡献大小反例当训练主导已证。")
claim(31,"will reduce OOD forbidden-contact rates while leaving perfect ID guards unchanged.","untrained_or_approximate_hypothesis_not_hard_error","explicit Hypothesis about future evaluation guards","此句在Hypothesis内，后续还有非回归容差预测；不把尚未训练的ID保持假说按既成事实或代码恒等保证硬判。所有八个父项确实保持。")
claim(32,"without disturbing the saturated in-distribution strata, because on near-flat support the stance-relative lift stays small.","untrained_or_approximate_hypothesis_not_hard_error","future ID behavior under near-flat support hypothesis","处于Hypothesis/should help语境，small没有定义为零；ID不变需实验，不当作代码恒等或已观察结果。父八项保持。最低stance的实现语义另属一般代码描述类别，不在本次保留/生效范围复核扩评。")


rules_path=ROOT/"R0B_REVIEW_RULES_ZH.md"
rules=rules_path.read_bytes()
results=[]
for i in range(1,33):
    r=records[i]["result"]
    statuses={c["status"] for c in r["claims"]}
    r["category_result"]=("algebraic_contradiction_found" if "algebraic_contradiction" in statuses else
                          "scope_or_evidence_limit_no_unambiguous_contradiction" if any(s in statuses for s in ("scope_ambiguity_not_hard_error","unsupported_historical_preservation_guarantee")) else
                          "no_contradiction_established_within_this_category")
    r["not_a_global_zero_error_verdict"]=True
    r["final_AR_or_mechanism_or_gate_verdict"]=None
    results.append(r)

payload={
    "schema_version":"p3r.posthoc_preservation_category_audit.v1",
    "audit_date":"2026-09-29",
    "epistemic_status":"事后限定类别一致性复核；不是新确认实验、不是新的独立盲评、不是重新计算历史分数。",
    "reviewer_origin":"Codex collaborative subagent /root/r0_preservation_claim_audit; exact backend model version receipt not available",
    "scope":"32项统一复核保留区域/机制、条件生效、完整保留或降低成本声明。其它数值、观察命题、训练假说收益不重评。",
    "uniform_rules":[
        "区分精确零点、数值近似和整个stable/dwell区间；nominal没有自动扩成整个区间。",
        "完整保持需函数及权重clip一致；边际斜率下降不等于总成本下降。",
        "soft gate趋近0不等于严格inactive。",
        "未定义upright/near/strong/about等范围的模糊措辞不强制硬判。",
        "无失败标签不自动推出所有物理force等于0；允许反驳推导但不编造历史轨迹反例。",
        "未来训练假说保持待验证。CPU输入仅反驳无条件代数命题，不证明已发生轨迹或因果。",
        "不将局部无矛盾写成全部硬错误0，不给最终行动相关性、机制对应或出口结论。"
    ],
    "read_boundaries":{"reviewer_files_read":128,"proposal_count":32,
                       "withhold_mapping_read":False,"historical_scores_or_aggregate_read":False,"audit_or_rescore_files_read":False,
                       "api_calls":False,"simulator_or_training_launched":False,
                       "current_truth_limit":"CodeGraph was attempted three times before production lookup but did not return the named Stair symbols. Narrow fallback reads covered reward_inputs.py:229-246, episode_evidence.py:507-545 and threshold search in task configs. These are current truth, never attributed as author-visible history."},
    "rules_file":{"file":rules_path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(rules).hexdigest()},
    "cpu_method":{"implementation":"NumPy on CPU, direct original candidate callable bodies with a minimal Tensor adapter; production imports omitted",
                  "counterexample_assertions_passed":True,"simulation":False,"script":Path(__file__).name},
    "truth_refs":[
        {"file":"results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer/proposal-001-seen.json","json_pointer":"/public_feedback_summary/public_dev/behavior/measurement_contract","use":"Stable limits and dwell length; 007 did not receive this detailed process contract, so this is external disconfirming context, not retroactive author evidence."},
        {"file":"results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer/proposal-002-seen.json","json_pointer":"/reward_inputs_contract/field_semantics_constraints/projected_gravity","use":"Euler/projected-gravity relation for qualified 006/026 analysis."},
        {"file":"results/ceiling_final_evidence_r1/ceiling_review_all_r1/reviewer/proposal-028-seen.json","json_pointer":"/public_feedback_summary/behavior/profile/semantics/thresholds/forbidden_N","value":1.0,"use":"Historical author-visible threshold in 028; not automatically author-visible in 015."},
        {"file":"configs/tasks/stair_go2_12dof_unified_protocol_v3.json","line":75,"value":1.0,"use":"Current cross-check of positive forbidden force threshold; not historical score evidence."}
    ],
    "summary":{"category_result_counts":dict(Counter(r["category_result"] for r in results)),
               "unambiguous_algebraic_contradiction_proposals":[r["proposal_id"] for r in results if r["category_result"]=="algebraic_contradiction_found"],
               "final_gate_decision":None,"historic_scores_modified":False},
    "proposals":results,"input_inventory":inventory
}
OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"output":str(OUT),"summary":payload["summary"],"files":len(inventory)},ensure_ascii=False,indent=2))
