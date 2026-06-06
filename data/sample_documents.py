"""
data/sample_documents.py
========================
Synthetic but realistic sample documents for the Agentic RAG pipeline.

30 documents are provided across 5 domains:
    - healthcare   (docs 001-006)
    - finance      (docs 007-012)
    - education    (docs 013-018)
    - law          (docs 019-024)
    - technology   (docs 025-030)

Each document is a dict with the keys:
    id       : str  -- unique identifier (e.g. 'doc_001')
    text     : str  -- 3-5 sentence body paragraph
    domain   : str  -- one of the five domain labels
    title    : str  -- short descriptive title
    metadata : dict -- source, year, and optional extra fields

Also provided:
    SAMPLE_QUERIES  : list[str]  -- 20 diverse queries (simple, complex, multi-hop)
    GROUND_TRUTH    : dict[str, list[str]]  -- expected answer keywords per query
"""

# ============================================================
# DOCUMENTS
# ============================================================

DOCUMENTS = [

    # ----------------------------------------------------------
    # HEALTHCARE (001-006)
    # ----------------------------------------------------------
    {
        "id": "doc_001",
        "title": "Type 2 Diabetes: Pathophysiology and Risk Factors",
        "domain": "healthcare",
        "text": (
            "Type 2 diabetes mellitus is a chronic metabolic disorder characterised by "
            "insulin resistance and progressive beta-cell dysfunction. Excess visceral "
            "adipose tissue releases pro-inflammatory cytokines that impair insulin "
            "signalling in skeletal muscle and liver, leading to hyperglycaemia. "
            "Key risk factors include obesity, physical inactivity, a high-glycaemic diet, "
            "and a family history of diabetes. Early detection through fasting plasma "
            "glucose or HbA1c screening is essential for preventing complications. "
            "Lifestyle modification including a Mediterranean-style diet and at least "
            "150 minutes of aerobic exercise per week remains the cornerstone of prevention."
        ),
        "metadata": {"source": "endocrinology_textbook", "year": 2023},
    },
    {
        "id": "doc_002",
        "title": "Metformin: Mechanism of Action and Clinical Use",
        "domain": "healthcare",
        "text": (
            "Metformin is a biguanide drug and the first-line pharmacological treatment "
            "for type 2 diabetes in most international guidelines. It primarily lowers "
            "blood glucose by inhibiting hepatic gluconeogenesis via activation of "
            "AMP-activated protein kinase (AMPK). Secondary benefits include modest "
            "weight reduction and a favourable cardiovascular risk profile, making it "
            "preferable to sulfonylureas in overweight patients. Common side effects are "
            "gastrointestinal in nature including nausea, diarrhoea, and abdominal "
            "discomfort and are minimised by dose titration with meals. Metformin is "
            "contraindicated in patients with severe renal impairment (eGFR < 30 mL/min) "
            "due to the risk of lactic acidosis."
        ),
        "metadata": {"source": "pharmacology_review", "year": 2022},
    },
    {
        "id": "doc_003",
        "title": "Insulin Therapy in Type 1 Diabetes",
        "domain": "healthcare",
        "text": (
            "Type 1 diabetes mellitus is an autoimmune condition in which the immune system "
            "destroys pancreatic beta cells, leading to absolute insulin deficiency. "
            "Patients require lifelong exogenous insulin therapy to maintain glycaemic "
            "control and prevent diabetic ketoacidosis (DKA). Modern management uses "
            "basal-bolus regimens combining long-acting insulin (e.g. insulin glargine) "
            "with rapid-acting analogues (e.g. insulin aspart) timed to meals. Continuous "
            "glucose monitoring (CGM) devices and closed-loop insulin delivery systems "
            "(artificial pancreas) represent state-of-the-art technology that significantly "
            "improves time-in-range metrics. Patients and caregivers must receive "
            "comprehensive diabetes education including carbohydrate counting and "
            "sick-day management."
        ),
        "metadata": {"source": "clinical_endocrinology_journal", "year": 2023},
    },
    {
        "id": "doc_004",
        "title": "Hypertension Management: Lifestyle and Drug Therapy",
        "domain": "healthcare",
        "text": (
            "Hypertension, defined as sustained blood pressure above 130/80 mmHg, affects "
            "approximately 1.28 billion adults worldwide and is a leading cause of "
            "cardiovascular disease and stroke. Non-pharmacological interventions such as "
            "the DASH diet, sodium restriction to under 2.3 g per day, regular aerobic "
            "exercise, and weight loss can lower systolic blood pressure by 5-10 mmHg. "
            "When lifestyle changes are insufficient, antihypertensive drugs are indicated; "
            "first-line options include ACE inhibitors, angiotensin receptor blockers (ARBs), "
            "calcium channel blockers, and thiazide diuretics. Combination therapy is "
            "frequently required, and adherence to medication is the single most important "
            "predictor of sustained blood pressure control."
        ),
        "metadata": {"source": "cardiology_guidelines_2023", "year": 2023},
    },
    {
        "id": "doc_005",
        "title": "Cholesterol and Statin Therapy",
        "domain": "healthcare",
        "text": (
            "Elevated low-density lipoprotein cholesterol (LDL-C) is a well-established "
            "causal risk factor for atherosclerotic cardiovascular disease (ASCVD). "
            "HMG-CoA reductase inhibitors, commonly known as statins, are the most widely "
            "prescribed lipid-lowering agents and reduce LDL-C by 30-55% depending on the "
            "drug and dose. High-intensity statins such as atorvastatin 40-80 mg and "
            "rosuvastatin 20-40 mg are recommended for patients with established ASCVD or "
            "high 10-year cardiovascular risk. Adverse effects including myopathy and "
            "elevated liver transaminases are uncommon but require monitoring. For patients "
            "who are statin-intolerant, PCSK9 inhibitors (e.g. evolocumab) offer an "
            "alternative with even greater LDL-C reduction."
        ),
        "metadata": {"source": "lipid_management_guidelines", "year": 2022},
    },
    {
        "id": "doc_006",
        "title": "Oncology: Immunotherapy and Checkpoint Inhibitors",
        "domain": "healthcare",
        "text": (
            "Cancer immunotherapy harnesses the body's own immune system to recognise "
            "and destroy malignant cells. Immune checkpoint inhibitors (ICIs) block "
            "inhibitory receptors most notably PD-1, PD-L1, and CTLA-4 that tumour cells "
            "exploit to evade immune surveillance. Pembrolizumab and nivolumab (PD-1 "
            "inhibitors) have transformed the treatment landscape for melanoma, non-small "
            "cell lung cancer, and several other solid tumours. Response rates correlate "
            "with tumour mutational burden (TMB) and PD-L1 expression, though biomarker "
            "selection remains an active research area. Immune-related adverse events "
            "(irAEs) including colitis, pneumonitis, and endocrinopathies require prompt "
            "recognition and may necessitate corticosteroid treatment."
        ),
        "metadata": {"source": "oncology_annual_review", "year": 2023},
    },

    # ----------------------------------------------------------
    # FINANCE (007-012)
    # ----------------------------------------------------------
    {
        "id": "doc_007",
        "title": "Stock Market Fundamentals: Equity Valuation",
        "domain": "finance",
        "text": (
            "Equity valuation is the process of estimating the intrinsic value of a "
            "company's shares to determine whether they are overpriced, fairly priced, "
            "or underpriced relative to the current market price. The discounted cash "
            "flow (DCF) model projects future free cash flows and discounts them back "
            "to the present using the weighted average cost of capital (WACC). "
            "Comparable company analysis benchmarks valuation multiples such as "
            "price-to-earnings (P/E), enterprise value-to-EBITDA (EV/EBITDA), and "
            "price-to-book (P/B) against industry peers. Dividend discount models (DDMs) "
            "are particularly relevant for mature, dividend-paying companies with "
            "predictable cash distributions. Combining multiple approaches reduces "
            "valuation risk and produces a more robust price range."
        ),
        "metadata": {"source": "financial_analysis_textbook", "year": 2023},
    },
    {
        "id": "doc_008",
        "title": "Portfolio Diversification and Modern Portfolio Theory",
        "domain": "finance",
        "text": (
            "Modern Portfolio Theory (MPT), introduced by Harry Markowitz in 1952, "
            "formalises the relationship between risk and expected return for a portfolio "
            "of financial assets. The efficient frontier represents the set of optimal "
            "portfolios that offer the highest expected return for a given level of risk, "
            "measured by standard deviation. Diversification reduces unsystematic "
            "company-specific risk without sacrificing expected return, because asset "
            "returns are not perfectly correlated. The capital asset pricing model (CAPM) "
            "extends MPT by introducing the concept of beta, which measures an asset's "
            "sensitivity to broad market movements. Investors are compensated only for "
            "systematic risk through the equity risk premium."
        ),
        "metadata": {"source": "investment_management_textbook", "year": 2022},
    },
    {
        "id": "doc_009",
        "title": "Interest Rates, Bonds, and Fixed Income Markets",
        "domain": "finance",
        "text": (
            "A bond is a debt instrument through which an issuer (government or corporation) "
            "borrows capital from investors and promises periodic coupon payments plus "
            "repayment of principal at maturity. Bond prices and yields move inversely: "
            "when interest rates rise, existing bond prices fall because newer issues offer "
            "more attractive coupons. Duration measures a bond's price sensitivity to "
            "interest rate changes; a bond with a 7-year modified duration loses approximately "
            "7% of its value for every 1% rise in yields. Credit ratings assigned by agencies "
            "such as Moody's, S&P, and Fitch reflect the issuer's default risk. "
            "Investment-grade bonds (BBB- or above) typically carry lower yields than "
            "high-yield bonds due to their lower default probability."
        ),
        "metadata": {"source": "fixed_income_analysis", "year": 2023},
    },
    {
        "id": "doc_010",
        "title": "Inflation, Central Banks, and Monetary Policy",
        "domain": "finance",
        "text": (
            "Inflation represents the rate at which the general price level of goods and "
            "services increases over time, eroding the purchasing power of money. Central "
            "banks, such as the US Federal Reserve and the European Central Bank, use "
            "monetary policy tools to maintain price stability, typically targeting annual "
            "inflation around 2%. The primary instrument is the policy interest rate; "
            "raising rates increases borrowing costs across the economy, dampening demand "
            "and thereby reducing inflationary pressure. Quantitative easing (QE) is an "
            "unconventional tool where the central bank purchases long-term assets to "
            "inject liquidity when policy rates are near zero. Persistent high inflation "
            "can entrench wage-price spirals and severely distort capital allocation."
        ),
        "metadata": {"source": "macroeconomics_textbook", "year": 2023},
    },
    {
        "id": "doc_011",
        "title": "Cryptocurrency and Blockchain Finance",
        "domain": "finance",
        "text": (
            "Bitcoin, launched in 2009 by the pseudonymous Satoshi Nakamoto, was the first "
            "decentralised cryptocurrency and introduced the concept of a distributed ledger "
            "secured by cryptographic proof-of-work. Blockchain technology records "
            "transactions in immutable, chronologically ordered blocks that are validated "
            "by a peer-to-peer network without the need for a central authority. Ethereum "
            "extended the blockchain paradigm by introducing smart contracts, which are "
            "self-executing code that automates financial agreements, underpinning "
            "decentralised finance (DeFi) applications. Regulatory uncertainty remains a "
            "significant risk, with major jurisdictions adopting divergent approaches "
            "ranging from outright bans to comprehensive licensing frameworks. High price "
            "volatility makes cryptocurrencies poorly suited as a stable medium of exchange."
        ),
        "metadata": {"source": "digital_assets_primer", "year": 2023},
    },
    {
        "id": "doc_012",
        "title": "Financial Risk Management: VaR and Stress Testing",
        "domain": "finance",
        "text": (
            "Value at Risk (VaR) is a widely used risk measure that quantifies the maximum "
            "potential loss on a portfolio over a specified time horizon at a given "
            "confidence level. A 1-day 99% VaR of $1 million means there is a 1% chance "
            "of losing more than $1 million in a single trading day. While VaR is "
            "intuitive, it underestimates tail risk in non-normal distributions and fails "
            "to capture losses beyond the confidence threshold. Expected Shortfall (ES), "
            "or Conditional VaR, addresses this by averaging losses in the worst scenarios "
            "beyond the VaR cutoff. Regulatory stress testing such as EBA stress tests in "
            "Europe and DFAST in the US subjects banks to hypothetical severe macroeconomic "
            "scenarios to assess capital adequacy. Robust risk management combines "
            "quantitative models with qualitative scenario analysis and expert judgment."
        ),
        "metadata": {"source": "risk_management_handbook", "year": 2022},
    },

    # ----------------------------------------------------------
    # EDUCATION (013-018)
    # ----------------------------------------------------------
    {
        "id": "doc_013",
        "title": "Machine Learning in Education: Personalised Learning",
        "domain": "education",
        "text": (
            "Adaptive learning platforms use machine learning algorithms to tailor "
            "educational content to individual students based on their prior knowledge, "
            "learning pace, and error patterns. Collaborative filtering models similar "
            "to those used in recommendation systems predict which learning resources a "
            "student is most likely to benefit from, drawing on the behaviour of similar "
            "learners. Knowledge tracing algorithms such as Bayesian Knowledge Tracing "
            "(BKT) and Deep Knowledge Tracing (DKT) model how a student's mastery of a "
            "concept evolves over time. Personalised learning has been shown to improve "
            "learning outcomes by 30% in controlled studies compared to traditional "
            "one-size-fits-all instruction. The main challenge is ensuring that algorithms "
            "do not perpetuate or amplify existing educational inequalities."
        ),
        "metadata": {"source": "educational_technology_journal", "year": 2023},
    },
    {
        "id": "doc_014",
        "title": "Transformer Models and Natural Language Processing",
        "domain": "education",
        "text": (
            "The Transformer architecture, introduced by Vaswani et al. in 2017, "
            "revolutionised natural language processing by replacing recurrent networks "
            "with a self-attention mechanism that captures long-range dependencies in "
            "parallel across the entire sequence. BERT (Bidirectional Encoder "
            "Representations from Transformers) pre-trains on masked language modelling "
            "and next-sentence prediction, producing contextual embeddings that achieve "
            "state-of-the-art results on a wide range of NLP benchmarks. GPT-series "
            "models use an autoregressive decoder-only design for generative tasks, "
            "culminating in GPT-4 with trillion-scale parameters and emergent few-shot "
            "capabilities. Fine-tuning pre-trained Transformers on domain-specific "
            "datasets requires far less labelled data than training from scratch, making "
            "the approach highly practical for specialised applications."
        ),
        "metadata": {"source": "deep_learning_textbook", "year": 2023},
    },
    {
        "id": "doc_015",
        "title": "Constructivism and Active Learning Pedagogies",
        "domain": "education",
        "text": (
            "Constructivism, rooted in the work of Piaget and Vygotsky, holds that "
            "learners build knowledge by actively constructing meaning through experience "
            "rather than passively receiving information. Active learning strategies "
            "including problem-based learning (PBL), flipped classrooms, and collaborative "
            "group work align with constructivist principles and have strong empirical "
            "support. Meta-analyses consistently show that active learning increases exam "
            "scores by roughly half a standard deviation compared to traditional lecturing. "
            "Spaced repetition and retrieval practice (the testing effect) are "
            "evidence-based techniques for improving long-term memory retention. "
            "Vygotsky's Zone of Proximal Development underscores the importance of "
            "providing appropriately scaffolded challenges just beyond a student's current "
            "competence."
        ),
        "metadata": {"source": "educational_psychology_textbook", "year": 2022},
    },
    {
        "id": "doc_016",
        "title": "Massive Open Online Courses (MOOCs) and Completion Rates",
        "domain": "education",
        "text": (
            "Massive Open Online Courses (MOOCs) emerged around 2012 with platforms such "
            "as Coursera, edX, and Udacity offering university-level content to millions "
            "of learners worldwide at no or low cost. Despite high enrolment numbers, "
            "completion rates for MOOCs are typically below 10%, a phenomenon attributed "
            "to low accountability, lack of social interaction, and mismatched learner "
            "expectations. Certificate programmes with verified assessments, peer-reviewed "
            "assignments, and discussion forums have meaningfully higher completion rates. "
            "Employer recognition of MOOC certificates is growing, particularly in "
            "technology fields such as data science and cloud computing. Research suggests "
            "that learners who already hold a degree are most likely to complete MOOCs, "
            "raising concerns about their impact on educational equity."
        ),
        "metadata": {"source": "online_education_research", "year": 2023},
    },
    {
        "id": "doc_017",
        "title": "STEM Education: Closing the Gender Gap",
        "domain": "education",
        "text": (
            "Despite significant progress over the past two decades, women remain "
            "underrepresented in science, technology, engineering, and mathematics (STEM) "
            "disciplines, particularly in computing and engineering. Stereotype threat, "
            "the fear of confirming negative group stereotypes, has been shown "
            "experimentally to impair the test performance of women in mathematics "
            "contexts. Interventions that highlight growth mindset, provide near-peer "
            "role models, and create inclusive classroom environments significantly reduce "
            "the gender attainment gap. Countries with greater gender equality in society "
            "also tend to have more balanced STEM participation, suggesting that cultural "
            "factors are critical. Early engagement through coding clubs, robotics "
            "programmes, and mentorship schemes in secondary school is especially "
            "effective at sparking sustained interest in STEM careers."
        ),
        "metadata": {"source": "gender_in_stem_report", "year": 2023},
    },
    {
        "id": "doc_018",
        "title": "Assessment Design: Formative vs Summative Evaluation",
        "domain": "education",
        "text": (
            "Assessment in education is broadly classified as formative or summative: "
            "formative assessment occurs during the learning process to provide feedback "
            "and guide instruction, while summative assessment evaluates learning at the "
            "end of an instructional period. Research by Black and Wiliam (1998) "
            "demonstrated that high-quality formative feedback is one of the most "
            "powerful interventions for raising student achievement. Rubrics, learning "
            "portfolios, and peer assessment are examples of formative tools that develop "
            "metacognitive skills alongside content knowledge. Summative instruments such "
            "as standardised tests enable comparability across institutions and "
            "jurisdictions but may narrow teaching to teaching to the test. Authentic "
            "assessment tasks such as projects, presentations, and simulations better "
            "reflect real-world competencies than traditional multiple-choice examinations."
        ),
        "metadata": {"source": "assessment_and_evaluation_handbook", "year": 2022},
    },

    # ----------------------------------------------------------
    # LAW (019-024)
    # ----------------------------------------------------------
    {
        "id": "doc_019",
        "title": "Contract Law: Formation and Enforceability",
        "domain": "law",
        "text": (
            "A legally binding contract requires four essential elements: offer, "
            "acceptance, consideration, and the intention to create legal relations. "
            "An offer is a definite proposal made by the offeror that, when accepted "
            "without modification, creates a binding agreement; any variation constitutes "
            "a counter-offer that terminates the original offer. Consideration, something "
            "of legal value exchanged between the parties, distinguishes a contract from "
            "a gratuitous promise. Contracts may be void, voidable, or unenforceable "
            "depending on factors such as misrepresentation, duress, undue influence, or "
            "illegality. Written contracts are required by the Statute of Frauds for "
            "certain categories, including the sale of land and guarantees."
        ),
        "metadata": {"source": "contract_law_textbook", "year": 2022},
    },
    {
        "id": "doc_020",
        "title": "Intellectual Property: Copyright and Patent Law",
        "domain": "law",
        "text": (
            "Intellectual property (IP) law protects the creations of the mind through "
            "distinct legal regimes: copyright protects original creative works for the "
            "life of the author plus 70 years in most jurisdictions, while patents grant "
            "inventors a 20-year exclusive right to commercialise a novel, non-obvious, "
            "and industrially applicable invention. Trademark law protects distinctive "
            "signs such as words, logos, or shapes that identify the source of goods or "
            "services and can theoretically last indefinitely with continued use and "
            "renewal. Trade secrets protect confidential business information such as "
            "formulas and processes without a fixed term, provided reasonable steps are "
            "taken to maintain secrecy. TRIPS (Trade-Related Aspects of Intellectual "
            "Property Rights) sets minimum IP standards for WTO member states, ensuring "
            "international harmonisation."
        ),
        "metadata": {"source": "ip_law_principles", "year": 2023},
    },
    {
        "id": "doc_021",
        "title": "Data Privacy Law: GDPR and Beyond",
        "domain": "law",
        "text": (
            "The General Data Protection Regulation (GDPR), effective May 2018, is the "
            "EU's flagship privacy legislation and imposes strict obligations on any "
            "organisation that processes personal data of EU residents, regardless of the "
            "organisation's geographic location. Key principles include lawfulness of "
            "processing, purpose limitation, data minimisation, accuracy, storage "
            "limitation, and integrity. Data subjects enjoy rights to access, "
            "rectification, erasure (right to be forgotten), portability, and objection "
            "to automated decision-making. Organisations must appoint a Data Protection "
            "Officer (DPO) if they carry out large-scale processing of sensitive "
            "categories of data. Fines under GDPR can reach EUR 20 million or 4% of "
            "global annual turnover, whichever is higher."
        ),
        "metadata": {"source": "data_privacy_law_review", "year": 2023},
    },
    {
        "id": "doc_022",
        "title": "Employment Law: Wrongful Dismissal and Employment Contracts",
        "domain": "law",
        "text": (
            "Employment law governs the relationship between employers and employees, "
            "balancing the employer's right to manage its workforce with the employee's "
            "right to fair treatment and job security. In most common-law jurisdictions, "
            "an at-will employment relationship can be terminated by either party without "
            "cause, but statutory protections prevent dismissal on discriminatory grounds "
            "such as race, gender, disability, age, or religion. Wrongful dismissal "
            "claims arise when an employee is terminated in breach of their employment "
            "contract, including situations where contractual notice periods are not "
            "honoured. Constructive dismissal occurs when an employer's conduct "
            "fundamentally breaches the employment contract, forcing the employee to "
            "resign. Remedies for unfair dismissal include reinstatement, re-engagement, "
            "and compensatory or basic awards."
        ),
        "metadata": {"source": "employment_law_handbook", "year": 2022},
    },
    {
        "id": "doc_023",
        "title": "Tort Law: Negligence and Duty of Care",
        "domain": "law",
        "text": (
            "The tort of negligence imposes civil liability on a defendant who breaches "
            "a duty of care owed to the claimant, causing foreseeable damage. The "
            "three-stage Caparo test asks whether damage was foreseeable, whether there "
            "was proximity between the parties, and whether it is fair, just, and "
            "reasonable to impose a duty. The standard of care is that of the reasonable "
            "person, though professionals are held to a higher standard such as the Bolam "
            "test in medical negligence. Causation requires the claimant to establish "
            "both factual causation (the but for test) and legal causation. Damages in "
            "negligence are compensatory, aiming to restore the claimant to their "
            "pre-tort position, and may include general damages (pain and suffering) and "
            "special damages (quantified financial losses)."
        ),
        "metadata": {"source": "tort_law_textbook", "year": 2023},
    },
    {
        "id": "doc_024",
        "title": "Company Law: Corporate Governance and Directors' Duties",
        "domain": "law",
        "text": (
            "Corporate governance refers to the system of rules, practices, and processes "
            "by which a company is directed and controlled, balancing the interests of "
            "shareholders, management, customers, suppliers, financiers, and the "
            "community. Directors owe fiduciary duties to the company, including the duty "
            "to act in good faith for the benefit of shareholders as a whole, to exercise "
            "independent judgment, and to avoid conflicts of interest. The business "
            "judgment rule protects directors from personal liability for decisions made "
            "in good faith, with due care, and in a rationally informed manner. "
            "Separation of the roles of Chairman and CEO, independent non-executive "
            "directors, and audit committees are governance best practices recommended by "
            "the UK Corporate Governance Code. Derivative actions allow shareholders to "
            "sue directors on the company's behalf for breaches of duty."
        ),
        "metadata": {"source": "company_law_and_governance", "year": 2023},
    },

    # ----------------------------------------------------------
    # TECHNOLOGY (025-030)
    # ----------------------------------------------------------
    {
        "id": "doc_025",
        "title": "Cloud Computing: IaaS, PaaS, and SaaS Models",
        "domain": "technology",
        "text": (
            "Cloud computing delivers on-demand computing resources including servers, "
            "storage, databases, networking, and software over the internet, typically "
            "on a pay-per-use basis. Infrastructure as a Service (IaaS) provides "
            "virtualised compute and storage such as AWS EC2 and Azure Virtual Machines, "
            "giving customers maximum control over the operating system and middleware. "
            "Platform as a Service (PaaS) abstracts the underlying infrastructure and "
            "offers a managed environment for application development and deployment, "
            "such as Google App Engine and Heroku. Software as a Service (SaaS) delivers "
            "fully managed applications over the web without any infrastructure management "
            "by the end user, examples being Salesforce and Microsoft 365. Hybrid and "
            "multi-cloud strategies balance cost, performance, compliance, and vendor "
            "lock-in risks."
        ),
        "metadata": {"source": "cloud_architecture_guide", "year": 2023},
    },
    {
        "id": "doc_026",
        "title": "Kubernetes: Container Orchestration at Scale",
        "domain": "technology",
        "text": (
            "Kubernetes (K8s) is an open-source container orchestration platform "
            "originally developed by Google and now maintained by the Cloud Native "
            "Computing Foundation (CNCF). It automates the deployment, scaling, and "
            "operation of containerised applications across clusters of physical or "
            "virtual machines. Core abstractions include Pods (the smallest deployable "
            "unit), Deployments (declarative updates for Pods), Services (stable network "
            "endpoints), and ConfigMaps/Secrets (configuration and sensitive data "
            "management). Horizontal Pod Autoscaling (HPA) dynamically adjusts the "
            "number of running replicas based on CPU or custom metrics, enabling elastic "
            "scalability. Kubernetes has become the de facto standard for running "
            "microservices architectures in production, supporting multi-cloud and "
            "on-premise deployments through standardised APIs."
        ),
        "metadata": {"source": "devops_engineering_handbook", "year": 2023},
    },
    {
        "id": "doc_027",
        "title": "Large Language Models: Architecture and Training",
        "domain": "technology",
        "text": (
            "Large Language Models (LLMs) are neural networks with billions to trillions "
            "of parameters trained on massive corpora of text data using self-supervised "
            "objectives such as next-token prediction. The Transformer architecture "
            "underpins most modern LLMs, with multi-head self-attention enabling efficient "
            "modelling of long-range token dependencies. Pre-training on internet-scale "
            "data imbues models with broad world knowledge and linguistic competence, "
            "while fine-tuning and Reinforcement Learning from Human Feedback (RLHF) "
            "align model behaviour with user intent and safety constraints. Emergent "
            "capabilities such as few-shot reasoning, code generation, and chain-of-"
            "thought problem-solving appear abruptly at certain scale thresholds. "
            "Challenges include hallucination, biased outputs, high inference latency, "
            "and enormous energy costs during training."
        ),
        "metadata": {"source": "ai_systems_overview", "year": 2023},
    },
    {
        "id": "doc_028",
        "title": "Cybersecurity: Zero Trust Architecture",
        "domain": "technology",
        "text": (
            "Zero Trust is a cybersecurity paradigm based on the principle of never "
            "trust, always verify, replacing the traditional perimeter-based security "
            "model that implicitly trusted anyone inside the corporate network. Every "
            "access request from users, devices, or services must be authenticated, "
            "authorised, and continuously validated regardless of its origin. Core "
            "components include strong identity verification via multi-factor "
            "authentication, least-privilege access control, micro-segmentation of the "
            "network, and comprehensive activity logging for anomaly detection. Zero "
            "Trust aligns with the modern enterprise reality of remote work, cloud "
            "services, and mobile devices that dissolve traditional network boundaries. "
            "Implementation typically requires a phased approach starting with identity "
            "and access management (IAM) before extending to device and workload security."
        ),
        "metadata": {"source": "enterprise_security_guide", "year": 2023},
    },
    {
        "id": "doc_029",
        "title": "Retrieval-Augmented Generation (RAG) Systems",
        "domain": "technology",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines the parametric knowledge of "
            "a large language model with non-parametric retrieval of relevant documents "
            "from an external corpus, addressing the hallucination and knowledge-cutoff "
            "limitations of standalone LLMs. At inference time, a retriever (typically a "
            "dense embedding model) identifies the most semantically similar documents to "
            "the user query, which are then provided as context in the LLM's prompt. "
            "Advanced RAG variants include iterative retrieval, query decomposition, "
            "hypothetical document embeddings (HyDE), and re-ranking with a cross-encoder. "
            "Agentic RAG extends this by allowing an LLM-based agent to autonomously "
            "plan multi-step retrieval strategies, use tools, and synthesise information "
            "from heterogeneous sources including vector stores, knowledge graphs, and "
            "APIs. Evaluation metrics include faithfulness, answer relevance, and context "
            "precision as defined in the RAGAs framework."
        ),
        "metadata": {"source": "rag_architecture_survey", "year": 2024},
    },
    {
        "id": "doc_030",
        "title": "DevOps: CI/CD Pipelines and Continuous Delivery",
        "domain": "technology",
        "text": (
            "DevOps is a cultural and technical movement that brings together software "
            "development (Dev) and IT operations (Ops) to shorten the software delivery "
            "lifecycle and improve deployment frequency, lead time, change failure rate, "
            "and mean time to recovery (MTTR), the four DORA metrics. Continuous "
            "Integration (CI) practices require developers to merge code changes to a "
            "shared repository multiple times per day, with each push triggering "
            "automated build and test pipelines. Continuous Delivery (CD) extends CI by "
            "ensuring that every code change that passes automated tests is deployable "
            "to production on demand. Infrastructure as Code (IaC) tools such as "
            "Terraform and Ansible enable declarative, version-controlled management of "
            "cloud infrastructure, reducing manual configuration drift. GitOps extends "
            "IaC principles by using Git as the single source of truth for both "
            "application and infrastructure state."
        ),
        "metadata": {"source": "devops_accelerate_research", "year": 2023},
    },
]

# ============================================================
# SAMPLE QUERIES
# ============================================================

SAMPLE_QUERIES: list = [
    # Simple factual
    "What is metformin used for?",
    "How does a FAISS vector store work?",
    "What are the four elements required to form a valid contract?",
    "What does the abbreviation GDPR stand for and what does it regulate?",
    "What is the difference between IaaS, PaaS, and SaaS?",

    # Descriptive / explanatory
    "Explain how statins lower LDL cholesterol.",
    "How do Transformer models use self-attention to understand language?",
    "Describe the key principles of Modern Portfolio Theory.",
    "What is Zero Trust architecture and why is it important?",
    "How does Retrieval-Augmented Generation address LLM hallucination?",

    # Comparative
    "Compare formative and summative assessment in education.",
    "What are the differences between type 1 and type 2 diabetes?",
    "How does copyright protection differ from patent protection?",
    "What distinguishes wrongful dismissal from constructive dismissal?",

    # Complex / multi-step reasoning
    (
        "How might rising interest rates simultaneously affect bond prices, "
        "equity valuations, and corporate borrowing costs?"
    ),
    (
        "Describe the pathway from insulin resistance to type 2 diabetes and "
        "explain how metformin intervenes in this pathway."
    ),
    (
        "How can machine learning techniques used in personalised education be "
        "applied to improve a RAG retrieval pipeline?"
    ),

    # Multi-hop / graph-relational
    (
        "Which treatment is recommended for type 2 diabetes, and what is the "
        "mechanism by which that treatment works?"
    ),
    (
        "Which regulatory body enforces GDPR fines, and what is the maximum "
        "penalty that can be imposed on a company?"
    ),
    (
        "If a company's LLM-based product processes personal data of EU residents "
        "and is deployed on Kubernetes in the cloud, what legal and technical "
        "compliance requirements apply?"
    ),
]

# ============================================================
# GROUND TRUTH  (query -> list of expected answer keywords)
# ============================================================

GROUND_TRUTH: dict = {
    "What is metformin used for?": [
        "type 2 diabetes",
        "blood glucose",
        "biguanide",
        "hepatic gluconeogenesis",
        "insulin resistance",
    ],
    "How does a FAISS vector store work?": [
        "FAISS",
        "vector",
        "similarity search",
        "embeddings",
        "index",
        "cosine similarity",
    ],
    "What are the four elements required to form a valid contract?": [
        "offer",
        "acceptance",
        "consideration",
        "intention",
    ],
    "What does the abbreviation GDPR stand for and what does it regulate?": [
        "General Data Protection Regulation",
        "personal data",
        "EU",
        "privacy",
        "data subjects",
    ],
    "What is the difference between IaaS, PaaS, and SaaS?": [
        "infrastructure",
        "platform",
        "software",
        "cloud",
        "managed",
        "service",
    ],
    "Explain how statins lower LDL cholesterol.": [
        "HMG-CoA reductase",
        "LDL-C",
        "cholesterol",
        "cardiovascular",
        "statin",
        "atorvastatin",
    ],
    "How do Transformer models use self-attention to understand language?": [
        "self-attention",
        "Transformer",
        "BERT",
        "long-range",
        "embeddings",
        "parallel",
    ],
    "Describe the key principles of Modern Portfolio Theory.": [
        "efficient frontier",
        "diversification",
        "risk",
        "expected return",
        "CAPM",
        "beta",
        "Markowitz",
    ],
    "What is Zero Trust architecture and why is it important?": [
        "never trust always verify",
        "authentication",
        "least privilege",
        "micro-segmentation",
        "identity",
        "perimeter",
    ],
    "How does Retrieval-Augmented Generation address LLM hallucination?": [
        "retrieval",
        "external corpus",
        "context",
        "hallucination",
        "dense embedding",
        "knowledge cutoff",
    ],
    "Compare formative and summative assessment in education.": [
        "formative",
        "summative",
        "feedback",
        "learning",
        "standardised",
        "metacognitive",
    ],
    "What are the differences between type 1 and type 2 diabetes?": [
        "autoimmune",
        "insulin deficiency",
        "beta cells",
        "insulin resistance",
        "type 1",
        "type 2",
    ],
    "How does copyright protection differ from patent protection?": [
        "copyright",
        "patent",
        "creative works",
        "invention",
        "70 years",
        "20 years",
        "novel",
    ],
    "What distinguishes wrongful dismissal from constructive dismissal?": [
        "wrongful dismissal",
        "constructive dismissal",
        "breach",
        "employment contract",
        "resign",
        "termination",
    ],
    (
        "How might rising interest rates simultaneously affect bond prices, "
        "equity valuations, and corporate borrowing costs?"
    ): [
        "bond prices fall",
        "yields rise",
        "WACC",
        "discount rate",
        "equity valuation",
        "borrowing costs",
        "DCF",
    ],
    (
        "Describe the pathway from insulin resistance to type 2 diabetes and "
        "explain how metformin intervenes in this pathway."
    ): [
        "insulin resistance",
        "hyperglycaemia",
        "beta-cell dysfunction",
        "AMPK",
        "gluconeogenesis",
        "metformin",
        "hepatic",
    ],
    (
        "How can machine learning techniques used in personalised education be "
        "applied to improve a RAG retrieval pipeline?"
    ): [
        "collaborative filtering",
        "knowledge tracing",
        "personalised",
        "retrieval",
        "adaptive",
        "recommendation",
    ],
    (
        "Which treatment is recommended for type 2 diabetes, and what is the "
        "mechanism by which that treatment works?"
    ): [
        "metformin",
        "first-line",
        "AMPK",
        "gluconeogenesis",
        "blood glucose",
        "biguanide",
    ],
    (
        "Which regulatory body enforces GDPR fines, and what is the maximum "
        "penalty that can be imposed on a company?"
    ): [
        "Data Protection Officer",
        "EUR 20 million",
        "4%",
        "global annual turnover",
        "GDPR",
        "EU",
    ],
    (
        "If a company's LLM-based product processes personal data of EU residents "
        "and is deployed on Kubernetes in the cloud, what legal and technical "
        "compliance requirements apply?"
    ): [
        "GDPR",
        "personal data",
        "Kubernetes",
        "cloud",
        "data minimisation",
        "security",
        "compliance",
        "DPO",
    ],
}
