#include "big_two_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_RANKS 13
#define NUM_SUITS 4

// --- 輔助函式 (C 語言版本) ---

static int get_rank(int card_index) {
    return card_index % NUM_RANKS;
}

static int get_suit(int card_index) {
    return card_index / NUM_RANKS;
}

static int get_card_value(int card_index) {
    int rank = get_rank(card_index);
    int suit = get_suit(card_index);
    // 數值: rank * 4 + suit (點數優先，花色次之)
    return rank * NUM_SUITS + suit;
}

// 排序函式 (用於 qsort)
static int compare_ints(const void *a, const void *b) {
    return (*(const int *)a - *(const int *)b);
}

// --- 牌型評估 (C 語言版本) ---

HandEvaluation evaluate_hand_c(const int8_t hand_vector[DECK_SIZE]) {
    HandEvaluation result = {PASS, 0};
    int indices[DECK_SIZE];
    int count = 0;
    
    // 獲取牌的索引
    for (int i = 0; i < DECK_SIZE; i++) {
        if (hand_vector[i] == 1) {
            indices[count++] = i;
        }
    }
    
    if (count == 0) {
        return result;
    }
    
    // 獲取點數、花色和牌值
    int ranks[count];
    int suits[count];
    int values[count];
    int max_value = 0;
    
    for (int i = 0; i < count; i++) {
        ranks[i] = get_rank(indices[i]);
        suits[i] = get_suit(indices[i]);
        values[i] = get_card_value(indices[i]);
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }
    
    // 統計點數出現次數
    int rank_counts[NUM_RANKS] = {0};
    for (int i = 0; i < count; i++) {
        rank_counts[ranks[i]]++;
    }
    
    // 找出非零的計數
    int non_zero_counts[NUM_RANKS];
    int unique_ranks[NUM_RANKS];
    int nz_count = 0;
    for (int i = 0; i < NUM_RANKS; i++) {
        if (rank_counts[i] > 0) {
            non_zero_counts[nz_count] = rank_counts[i];
            unique_ranks[nz_count] = i;
            nz_count++;
        }
    }
    
    // 對計數進行排序 (降序)
    // 由於 C 語言的 qsort 只能升序，我們需要一個輔助陣列來儲存計數
    int counts_array[nz_count];
    for (int i = 0; i < nz_count; i++) {
        counts_array[i] = non_zero_counts[i];
    }
    qsort(counts_array, nz_count, sizeof(int), compare_ints);
    
    // --- 牌型判斷 ---
    
    if (count == 1) {
        result.type = SINGLE;
        result.value = max_value;
        return result;
    }
    
    if (count == 2) {
        if (nz_count == 1 && counts_array[0] == 2) {
            result.type = PAIR;
            // 對子的點數值 * 4 + 最大花色值
            result.value = ranks[0] * NUM_SUITS + max_value;
            return result;
        }
    }
    
    if (count == 3) {
        if (nz_count == 1 && counts_array[0] == 3) {
            result.type = TRIPLE;
            // 三條的點數值 * 4 + 最大花色值
            result.value = ranks[0] * NUM_SUITS + max_value;
            return result;
        }
    }
    
    if (count == 5) {
        int is_flush = 1;
        for (int i = 1; i < count; i++) {
            if (suits[i] != suits[0]) {
                is_flush = 0;
                break;
            }
        }
        
        // 檢查是否為順子
        int sorted_ranks[count];
        for (int i = 0; i < count; i++) {
            sorted_ranks[i] = ranks[i];
        }
        qsort(sorted_ranks, count, sizeof(int), compare_ints);
        
        int is_straight = 0;
        int straight_value = 0;
        
        // 檢查 A2345 (11, 12, 0, 1, 2)
        if (sorted_ranks[0] == 0 && sorted_ranks[1] == 1 && sorted_ranks[2] == 2 && sorted_ranks[3] == 11 && sorted_ranks[4] == 12) {
            is_straight = 1;
            straight_value = max_value; // 為了簡化，使用最大牌值
        } else {
            // 檢查一般順子
            is_straight = 1;
            for (int i = 1; i < count; i++) {
                if (sorted_ranks[i] != sorted_ranks[i-1] + 1) {
                    is_straight = 0;
                    break;
                }
            }
            if (is_straight) {
                straight_value = max_value;
            }
        }
        
        // 檢查葫蘆、鐵支
        if (nz_count == 2) {
            // counts_array 是升序，所以 counts_array[1] 是最大的計數
            if (counts_array[1] == 4 && counts_array[0] == 1) {
                // 鐵支
                result.type = FOUR_OF_A_KIND;
                int four_rank = -1;
                for (int i = 0; i < NUM_RANKS; i++) {
                    if (rank_counts[i] == 4) {
                        four_rank = i;
                        break;
                    }
                }
                // 鐵支的大小只由四張牌的點數決定
                result.value = four_rank * NUM_SUITS + 3; // 3 是最大的花色索引
                return result;
            }
            if (counts_array[1] == 3 && counts_array[0] == 2) {
                // 葫蘆
                result.type = FULL_HOUSE;
                int triple_rank = -1;
                for (int i = 0; i < NUM_RANKS; i++) {
                    if (rank_counts[i] == 3) {
                        triple_rank = i;
                        break;
                    }
                }
                // 葫蘆的大小只由三張牌的點數決定
                result.value = triple_rank * NUM_SUITS + 3; // 3 是最大的花色索引
                return result;
            }
        }
        
        // 順子、同花、同花順
        if (is_straight && is_flush) {
            result.type = STRAIGHT_FLUSH;
            result.value = straight_value;
            return result;
        } else if (is_flush) {
            result.type = FLUSH;
            result.value = max_value;
            return result;
        } else if (is_straight) {
            result.type = STRAIGHT;
            result.value = straight_value;
            return result;
        }
    }
    
    return result;
}

// --- 牌型比較 (C 語言版本) ---

int compare_hands_c(const int8_t hand1_vector[DECK_SIZE], const int8_t hand2_vector[DECK_SIZE]) {
    HandEvaluation eval1 = evaluate_hand_c(hand1_vector);
    HandEvaluation eval2 = evaluate_hand_c(hand2_vector);
    
    // 牌型不符
    if (eval1.type != eval2.type) {
        return 0;
    }
    
    // 牌型相同，比較大小
    if (eval1.value > eval2.value) {
        return 1;
    } else if (eval1.value < eval2.value) {
        return -1;
    } else {
        return 0;
    }
}

// --- 合法動作遮罩 (C 語言版本) ---

void get_action_mask_c(
    const int8_t hand_vector[DECK_SIZE],
    const int8_t last_played_hand_vector[DECK_SIZE],
    int is_start_of_game,
    const int8_t fixed_actions[][DECK_SIZE],
    int8_t mask[ACTION_SPACE_SIZE]
) {
    // 初始化遮罩
    memset(mask, 0, ACTION_SPACE_SIZE * sizeof(int8_t));
    
    // 檢查牌桌是否為空
    int last_played_count = 0;
    for (int i = 0; i < DECK_SIZE; i++) {
        if (last_played_hand_vector[i] == 1) {
            last_played_count++;
        }
    }
    
    // 1. Pass 總是合法的 (除非是新一輪且必須出牌)
    if (last_played_count > 0) {
        mask[0] = 1;
    }
    
    // 2. 檢查固定動作
    for (int i = 0; i < ACTION_SPACE_SIZE - 1; i++) {
        const int8_t *action_vector = fixed_actions[i];
        int action_index = i + 1;
        
        // 檢查手牌是否包含該動作 (action_vector <= hand_vector)
        int is_in_hand = 1;
        for (int j = 0; j < DECK_SIZE; j++) {
            if (action_vector[j] > hand_vector[j]) {
                is_in_hand = 0;
                break;
            }
        }
        if (!is_in_hand) {
            continue;
        }
        
        // 檢查是否符合出牌規則
        if (last_played_count == 0) {
            // 牌桌為空，任何牌都可以出
            // 遊戲開始時，必須出梅花 3 (索引 0)
            if (is_start_of_game && action_vector[0] == 0) {
                continue; // 動作中沒有梅花 3，非法
            }
            
            // 動作中包含梅花 3，合法
            if (is_start_of_game && action_vector[0] == 1) {
                mask[action_index] = 1;
            }
            
            // 非遊戲開始，任何牌都可以出
            if (!is_start_of_game) {
                mask[action_index] = 1;
            }
            
        } else {
            // 比較大小
            if (compare_hands_c(action_vector, last_played_hand_vector) == 1) {
                mask[action_index] = 1;
            }
        }
    }
    
    // 3. 如果沒有任何合法動作，則必須 Pass
    int total_mask_sum = 0;
    for (int i = 1; i < ACTION_SPACE_SIZE; i++) {
        total_mask_sum += mask[i];
    }
    
    if (total_mask_sum == 0 && last_played_count > 0) {
        mask[0] = 1;
    }
}
