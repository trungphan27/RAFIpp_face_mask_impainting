## 1. Sequence Diagram
```mermaid
sequenceDiagram
    actor User as Trader/User
    participant UI as Portfolio.jsx
    participant PS as portfolioStore.js
    participant SC as Supabase Client
    participant DB as Supabase Tables
    participant LS as leaderboardStore.js

    User->>UI: Buy/Sell stock
    UI->>PS: buy() / sell()
    PS->>PS: Update cash, holdings, transactions
    PS->>SC: get authenticated user
    SC->>DB: Upsert portfolios
    SC->>DB: Insert transactions
    PS->>LS: submitScore(portfolioValue, totalReturn, tradeCount)
    LS->>SC: get user profile
    SC->>DB: Select user_profiles
    LS->>SC: upsert leaderboard score
    SC->>DB: Upsert leaderboard_entries
    LS->>SC: fetch leaderboard
    SC->>DB: Select leaderboard_entries
    LS-->>UI: Updated entries and user rank
```

# Portfolio, Risk, and Leaderboard Component - UML Class Diagram

## UML Relationship Legend

```mermaid
classDiagram
    direction LR

    Association_Source --> Association_Target : Association
    Child_Class --|> Parent_Class : Inheritance
    Implementing_Class ..|> Interface_Class : Realization / Implementation
    Dependent_Class ..> Service_Class : Dependency
    Aggregate_Whole o-- Aggregate_Part : Aggregation
    Composite_Whole *-- Composite_Part : Composition
```

| UML relationship | Mermaid syntax | Meaning |
|---|---:|---|
| Association | `A --> B` | A has a stable structural/use relationship with B. |
| Inheritance | `Child --|> Parent` | Child extends Parent. |
| Realization / Implementation | `Class ..|> Interface` | Class implements Interface. |
| Dependency | `A ..> B` | A temporarily calls or depends on B's API. |
| Aggregation | `A o-- B` | A groups B, but B can exist independently. |
| Composition | `A *-- B` | A strongly owns B as part of its state/lifecycle. |

## Class Diagram

```mermaid
classDiagram
    direction LR

    namespace React_Page {
        class PortfolioPage {
            <<React Page>>
            +render()
            +downloadCSV()
        }

        class LeaderboardPage {
            <<React Page>>
            +render()
            +changePeriod(period)
        }
    }

    namespace Zustand_Store {
        class PortfolioStore {
            <<Zustand Store>>
            -cash
            -initialCash
            -holdings
            -transactions
            -portfolioHistory
            -dailyPLHistory
            +buy(ticker, quantity, price, orderType)
            +sell(ticker, quantity, price, orderType)
            +syncToSupabase(tx)
            +loadFromSupabase()
            +getPortfolioValue(prices)
            +getUnrealizedPL(prices)
            +getTotalRealizedPL()
            +getTotalReturn(prices)
            +getHoldingsArray(prices)
            +getAllocation(prices)
            +recordSnapshot(prices)
            +submitToLeaderboard(prices)
            +reset()
        }

        class LeaderboardStore {
            <<Zustand Store>>
            -entries
            -period
            -userRank
            -loaded
            +setPeriod(period)
            +setEntries(entries)
            +fetchFromSupabase()
            +submitScore(portfolioValue, totalReturn, tradesCount)
        }

        class MarketStore {
            <<Zustand Store>>
            -prices
            -prevPrices
            -rawTicks
            +getChange(ticker)
            +getOHLCV(ticker, timeframe)
            +updatePrices(data)
            +simulateTick()
        }

        class AuthStore {
            <<Zustand Store>>
            -user
            -session
        }
    }

    namespace External_Client {
        class SupabaseClient {
            <<External Client>>
            +auth.getUser()
            +from(table)
            +isSupabaseConfigured()
        }
    }

    namespace Express_Router {
        class PortfolioRoute {
            <<Express Router>>
            +GET /api/portfolio
            +GET /api/portfolio/history
            +GET /api/portfolio/risk
        }

        class LeaderboardRoute {
            <<Express Router>>
            +GET /api/leaderboard
        }
    }

    namespace Service {
        class RiskMetrics {
            <<Service>>
            +sharpeRatio(returns, riskFreeRate)
            +maxDrawdown(values)
            +volatility(returns, annualized)
            +beta(portfolioReturns, marketReturns)
            +winRate(trades)
            +profitFactor(trades)
        }

        class SimulationEngine {
            <<Service>>
            -stocks
            -prices
            -rawTicks
            -regime
            +tick()
            +start(intervalMs)
            +stop()
            +onTick(callback)
            +setRegime(regime, params)
            +applyShock(ticker, shockPercent)
            +getQuote(ticker)
        }
    }

    namespace Data_Model {
        class Portfolio {
            <<Data Model>>
            +userId
            +cash
            +initialCash
            +holdings
            +updatedAt
        }

        class Holding {
            <<Data Model>>
            +ticker
            +shares
            +avgPrice
            +realizedPL
        }

        class Transaction {
            <<Data Model>>
            +id
            +type
            +ticker
            +orderType
            +quantity
            +price
            +total
            +time
            +status
        }

        class LeaderboardEntry {
            <<Data Model>>
            +userId
            +displayName
            +portfolioValue
            +totalReturn
            +sharpeRatio
            +tradesCount
            +period
            +updatedAt
        }

        class UserProfile {
            <<Data Model>>
            +id
            +displayName
            +watchlist
        }
    }

    PortfolioPage --> PortfolioStore : Association - reads state
    PortfolioPage --> MarketStore : Association - reads prices
    LeaderboardPage --> LeaderboardStore : Association - reads rankings
    LeaderboardPage --> AuthStore : Association - checks user

    PortfolioStore "1" *-- "1" Portfolio : Composition - owns state
    Portfolio "1" *-- "0..*" Holding : Composition - contains
    Portfolio "1" *-- "0..*" Transaction : Composition - contains

    PortfolioStore ..> MarketStore : Dependency - price input
    PortfolioStore ..> SupabaseClient : Dependency - sync/load
    PortfolioStore ..> LeaderboardStore : Dependency - submit score

    LeaderboardStore "1" o-- "0..*" LeaderboardEntry : Aggregation - fetched entries
    LeaderboardStore ..> SupabaseClient : Dependency - fetch/upsert
    AuthStore ..> SupabaseClient : Dependency - auth session

    SupabaseClient --> Portfolio : Association - portfolios
    SupabaseClient --> Transaction : Association - transactions
    SupabaseClient --> LeaderboardEntry : Association - leaderboard_entries
    SupabaseClient --> UserProfile : Association - user_profiles

    LeaderboardEntry --> UserProfile : Association - user profile

    PortfolioRoute "1" o-- "1" Portfolio : Aggregation - in-memory state
    PortfolioRoute ..> SimulationEngine : Dependency - current prices
    PortfolioRoute ..> RiskMetrics : Dependency - risk metrics
    LeaderboardRoute "1" o-- "0..*" LeaderboardEntry : Aggregation - ranking rows
```

## Notes

- The current source code does not define inheritance or interface implementation relationships for this component, so those arrows are shown only in the legend.
- `Composition` is used where the object is part of the owning state, such as `Portfolio` containing `Holding` and `Transaction`.
- `Aggregation` is used where a class groups data that can exist independently, such as `LeaderboardStore` grouping `LeaderboardEntry`.
- `Dependency` is used for API/service calls, such as `PortfolioStore` calling `SupabaseClient` or `RiskMetrics`.
