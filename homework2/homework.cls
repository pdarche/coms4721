\LoadClass[12pt]{article}
\ProvidesClass{homework}

\RequirePackage{fancyhdr}
\RequirePackage[letterpaper,margin=1in]{geometry}

\newcommand{\studentname}[1]{\gdef\@studentname{#1}}
\newcommand{\studentmail}[1]{\gdef\@studentmail{#1}}
\newcommand{\coursename}[1]{\gdef\@coursename{#1}}
\newcommand{\homeworknumber}[1]{\gdef\@homeworknumber{#1}}
\newcommand{\collaborators}[1]{\gdef\@collaborators{#1}}

\let\Oldmaketitle\maketitle
\renewcommand{\maketitle}
{
    \Oldmaketitle
    \thispagestyle{plain}
}

\fancypagestyle{plain}
{
    \fancyhead{}
    \fancyfoot{}
    \fancyfoot[LE,LO]{\@studentname \ - \texttt{\@studentmail}}
    \fancyfoot[RE,RO]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

\pagestyle{fancy}

\fancyhead{}
\fancyfoot{}
\fancyhead[RE,RO]{Problem Set \#\@homeworknumber}
\fancyhead[LE,LO]{\@coursename}
\fancyfoot[RE,RO]{\thepage}
\fancyfoot[LE,LO]{\@studentname \ - \texttt{\@studentmail}}
\renewcommand{\footrulewidth}{0.3pt}

\title{\textsf{\@coursename: Homework \#\@homeworknumber}}
\author{%
  \@studentname \ - \texttt{\@studentmail} \\
  Discussants: \texttt{\@collaborators}%
}
\date{\today}

% optional macros
\RequirePackage{amsmath,amsbsy,amsfonts,amssymb,amsthm,dsfont}

\def\ddefloop#1{\ifx\ddefloop#1\else\ddef{#1}\expandafter\ddefloop\fi}

% \bfA, \bfB, ...
\def\ddef#1{\expandafter\def\csname bf#1\endcsname{\ensuremath{\mathbf{#1}}}}
\ddefloop ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\ddefloop

% \bfalpha, \bfbeta, ...,  \bfGamma, \bfDelta, ...,
\def\ddef#1{\expandafter\def\csname bf#1\endcsname{\ensuremath{\pmb{\csname #1\endcsname}}}}
\ddefloop {alpha}{beta}{gamma}{delta}{epsilon}{varepsilon}{zeta}{eta}{theta}{vartheta}{iota}{kappa}{lambda}{mu}{nu}{xi}{pi}{varpi}{rho}{varrho}{sigma}{varsigma}{tau}{upsilon}{phi}{varphi}{chi}{psi}{omega}{Gamma}{Delta}{Theta}{Lambda}{Xi}{Pi}{Sigma}{varSigma}{Upsilon}{Phi}{Psi}{Omega}{ell}\ddefloop

% \bbA, \bbB, ...
\def\ddef#1{\expandafter\def\csname bb#1\endcsname{\ensuremath{\mathbb{#1}}}}
\ddefloop ABCDEFGHIJKLMNOPQRSTUVWXYZ\ddefloop

% \cA, \cB, ...
\def\ddef#1{\expandafter\def\csname c#1\endcsname{\ensuremath{\mathcal{#1}}}}
\ddefloop ABCDEFGHIJKLMNOPQRSTUVWXYZ\ddefloop

% math operators
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\Diag}{Diag}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\sign}{sign}
\DeclareMathOperator{\supp}{supp}
\DeclareMathOperator{\vol}{vol}
\DeclareMathOperator{\var}{var}
\DeclareMathOperator{\cov}{cov}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}

% boldface command (e.g., \mbf1 for boldface 1)
\newcommand\mbf{\ensuremath{\mathbf}}

% grouping commands
\newcommand\parens[1]{(#1)}
\newcommand\norm[1]{\|#1\|}
\newcommand\braces[1]{\{#1\}}
\newcommand\brackets[1]{[#1]}
\newcommand\ceil[1]{\lceil#1\rceil}
\newcommand\abs[1]{|#1|}
\newcommand\ind[1]{\ensuremath{\mathds{1}\{#1\}}}
\newcommand\dotp[1]{\langle #1 \rangle}

% variable-height grouping commands
\newcommand\Parens[1]{\left(#1\right)}
\newcommand\Norm[1]{\left\|#1\right\|}
\newcommand\Braces[1]{\left\{#1\right\}}
\newcommand\Brackets[1]{\left[#1\right]}
\newcommand\Ceil[1]{\left\lceil#1\right\rceil}
\newcommand\Abs[1]{\left|#1\right|}
\newcommand\Ind[1]{\mathds{1}\left\{#1\right\}}
\newcommand\Dotp[1]{\left\langle#1\right\rangle}

% theorem styles
\newtheorem{lemma}{Lemma}
\newtheorem{claim}{Claim}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}{Corollary}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{condition}{Condition}
\newtheorem{example}{Example}
\newtheorem{problem}{Problem}

\endinput