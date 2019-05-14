abstract sig Activity {}
abstract sig Payload {}

abstract sig Event{
	task: one Activity,
	data: set Payload,
	tokens: set Token
}

one sig DummyPayload extends Payload {}
fact { no te:Event | DummyPayload in te.data }

one sig DummyActivity extends Activity {}

abstract sig Token {}
abstract sig SameToken extends Token {}
abstract sig DiffToken extends Token {}
lone sig DummySToken extends SameToken{}
lone sig DummyDToken extends DiffToken{}
fact { 
	no DummySToken
	no DummyDToken
	all te:Event| no (te.tokens & SameToken) or no (te.tokens & DiffToken)
}

pred True[]{some TE0}

// lang templates

pred Init(taskA: Activity) { 
	taskA = TE0.task
}

pred Existence(taskA: Activity) { 
	some te: Event | te.task = taskA
}

pred Existence(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } >= n
}

pred Absence(taskA: Activity) { 
	no te: Event | te.task = taskA
}

pred Absence(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } <= n
}

pred Exactly(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } = n
}

pred Choice(taskA, taskB: Activity) { 
	some te: Event | te.task = taskA or te.task = taskB
}

pred ExclusiveChoice(taskA, taskB: Activity) { 
	some te: Event | te.task = taskA or te.task = taskB
	(no te: Event | taskA = te.task) or (no te: Event | taskB = te.task )
}

pred RespondedExistence(taskA, taskB: Activity) {
	(some te: Event | taskA = te.task) implies (some ote: Event | taskB = ote.task)
}

pred Response(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[te, fte])
}

pred AlternateResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[te, fte] and (no ite: Event | taskA = ite.task and After[te, ite] and After[ite, fte]))
}

pred ChainResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and Next[te, fte])
}

pred Precedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[fte, te])
}

pred AlternatePrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[fte, te] and (no ite: Event | taskA = ite.task and After[fte, ite] and After[ite, te]))
}

pred ChainPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and Next[fte, te])
}

pred NotRespondedExistence(taskA, taskB: Activity) {
	(some te: Event | taskA = te.task) implies (no te: Event | taskB = te.task)
}

pred NotResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | taskB = fte.task and After[te, fte])
}

pred NotPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | taskB = fte.task and After[fte, te])
}

pred NotChainResponse(taskA, taskB: Activity) { 
	all te: Event | taskA = te.task implies (no fte: Event | (DummyActivity = fte.task or taskB = fte.task) and Next[te, fte])
}

pred NotChainPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | (DummyActivity = fte.task or taskB = fte.task) and Next[fte, te])
}
//-

pred example { }
run example

---------------------- end of static code block ----------------------

--------------------- generated code starts here ---------------------

one sig examine_patient extends Activity {}
one sig perform_X_ray extends Activity {}
one sig check_X_ray_risk extends Activity {}
one sig perform_reposition extends Activity {}
one sig apply_cast extends Activity {}
one sig remove_cast extends Activity {}
one sig perform_surgery extends Activity {}
one sig prescribe_rehabilitation extends Activity {}
one sig TE0 extends Event {}{not task=DummyActivity}
one sig TE1 extends Event {}{not task=DummyActivity}
one sig TE2 extends Event {}{not task=DummyActivity}
one sig TE3 extends Event {}{not task=DummyActivity}
one sig TE4 extends Event {}{not task=DummyActivity}
one sig TE5 extends Event {}{not task=DummyActivity}
one sig TE6 extends Event {}{not task=DummyActivity}
one sig TE7 extends Event {}{not task=DummyActivity}
one sig TE8 extends Event {}{not task=DummyActivity}
one sig TE9 extends Event {}{not task=DummyActivity}
one sig TE10 extends Event {}{not task=DummyActivity}
one sig TE11 extends Event {}{not task=DummyActivity}
one sig TE12 extends Event {}{not task=DummyActivity}
one sig TE13 extends Event {}{not task=DummyActivity}
one sig TE14 extends Event {}{not task=DummyActivity}
one sig TE15 extends Event {}{not task=DummyActivity}
one sig TE16 extends Event {}{not task=DummyActivity}
one sig TE17 extends Event {}{not task=DummyActivity}
one sig TE18 extends Event {}{not task=DummyActivity}
pred Next(pre, next: Event){pre=TE0 and next=TE1 or pre=TE1 and next=TE2 or pre=TE2 and next=TE3 or pre=TE3 and next=TE4 or pre=TE4 and next=TE5 or pre=TE5 and next=TE6 or pre=TE6 and next=TE7 or pre=TE7 and next=TE8 or pre=TE8 and next=TE9 or pre=TE9 and next=TE10 or pre=TE10 and next=TE11 or pre=TE11 and next=TE12 or pre=TE12 and next=TE13 or pre=TE13 and next=TE14 or pre=TE14 and next=TE15 or pre=TE15 and next=TE16 or pre=TE16 and next=TE17 or pre=TE17 and next=TE18}
pred After(b, a: Event){// b=before, a=after
b=TE0 or a=TE18 or b=TE1 and not (a=TE0) or b=TE2 and not (a=TE0 or a=TE1) or b=TE3 and not (a=TE0 or a=TE1 or a=TE2) or b=TE4 and not (a=TE0 or a=TE1 or a=TE2 or a=TE3) or b=TE5 and not (a=TE0 or a=TE1 or a=TE2 or a=TE3 or a=TE4) or b=TE6 and not (a=TE0 or a=TE1 or a=TE2 or a=TE3 or a=TE4 or a=TE5) or b=TE7 and not (a=TE0 or a=TE1 or a=TE2 or a=TE3 or a=TE4 or a=TE5 or a=TE6) or b=TE8 and not (a=TE0 or a=TE1 or a=TE2 or a=TE3 or a=TE4 or a=TE5 or a=TE6 or a=TE7) or b=TE9 and (a=TE18 or a=TE17 or a=TE16 or a=TE15 or a=TE14 or a=TE13 or a=TE12 or a=TE11 or a=TE10) or b=TE10 and (a=TE18 or a=TE17 or a=TE16 or a=TE15 or a=TE14 or a=TE13 or a=TE12 or a=TE11) or b=TE11 and (a=TE18 or a=TE17 or a=TE16 or a=TE15 or a=TE14 or a=TE13 or a=TE12) or b=TE12 and (a=TE18 or a=TE17 or a=TE16 or a=TE15 or a=TE14 or a=TE13) or b=TE13 and (a=TE18 or a=TE17 or a=TE16 or a=TE15 or a=TE14) or b=TE14 and (a=TE18 or a=TE17 or a=TE16 or a=TE15) or b=TE15 and (a=TE18 or a=TE17 or a=TE16) or b=TE16 and (a=TE18 or a=TE17)}
pred p100040(A: Event) { { True[] } }
pred p100041(A: Event) { { True[] } }
pred p100042(A: Event) { { True[] } }
pred p100043(A: Event) { { True[] } }
pred p100044(A: Event) { { True[] } }
pred p100045(A: Event) { { True[] } }
pred p100046(A: Event) { { True[] } }
pred p100047(A: Event) { { True[] } }
fact {
Init[check_X_ray_risk]
AlternatePrecedence[perform_X_ray,check_X_ray_risk]
Precedence[perform_reposition,perform_X_ray]
Precedence[apply_cast,perform_X_ray]
NotResponse[apply_cast,remove_cast]
Precedence[remove_cast,apply_cast]
NotResponse[perform_X_ray,perform_surgery]
Response[perform_surgery,prescribe_rehabilitation]
#{ te: Event | te.task = examine_patient and p100040[te]} <= 3
#{ te: Event | te.task = perform_X_ray and p100041[te]} <= 3
#{ te: Event | te.task = check_X_ray_risk and p100042[te]} <= 3
#{ te: Event | te.task = perform_reposition and p100043[te]} <= 3
#{ te: Event | te.task = apply_cast and p100044[te]} <= 3
#{ te: Event | te.task = remove_cast and p100045[te]} <= 3
#{ te: Event | te.task = perform_surgery and p100046[te]} <= 3
#{ te: Event | te.task = prescribe_rehabilitation and p100047[te]} <= 3
}
