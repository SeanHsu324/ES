#ifndef BIG_TWO_CORE_H
#define BIG_TWO_CORE_H

#include <stdint.h>

#define DECK_SIZE 52
#define ACTION_SPACE_SIZE 4551 // Pass + 4550 fixed actions

// 牌型定義
typedef enum {
    PASS = 0,
    SINGLE = 1,
    PAIR = 2,
    TRIPLE = 3,
    STRAIGHT = 4,
    FLUSH = 5,
    FULL_HOUSE = 6,
    FOUR_OF_A_KIND = 7,
    STRAIGHT_FLUSH = 8
} HandType;

// 牌型評估結果結構
typedef struct {
    int type;  // HandType
    int value; // 用於比較大小的單一整數值
} HandEvaluation;

// 函數聲明

/**
 * @brief 評估一手牌的牌型和大小。
 * 
 * @param hand_vector 52維的牌向量 (0或1)。
 * @return HandEvaluation 包含牌型和大小值的結構。
 */
HandEvaluation evaluate_hand_c(const int8_t hand_vector[DECK_SIZE]);

/**
 * @brief 比較兩手牌的大小。
 * 
 * @param hand1_vector 第一手牌的牌向量。
 * @param hand2_vector 第二手牌的牌向量。
 * @return int 1 (hand1 > hand2), 0 (hand1 == hand2 或牌型不符), -1 (hand1 < hand2)。
 */
int compare_hands_c(const int8_t hand1_vector[DECK_SIZE], const int8_t hand2_vector[DECK_SIZE]);

/**
 * @brief 生成合法動作遮罩。
 * 
 * @param hand_vector 當前玩家的手牌向量。
 * @param last_played_hand_vector 最後出的牌向量。
 * @param is_start_of_game 是否為遊戲開始 (必須出梅花 3)。
 * @param fixed_actions 預先計算好的固定動作列表 (ACTION_SPACE_SIZE - 1 個動作)。
 * @param mask 輸出參數，用於儲存合法動作遮罩 (ACTION_SPACE_SIZE)。
 */
void get_action_mask_c(
    const int8_t hand_vector[DECK_SIZE],
    const int8_t last_played_hand_vector[DECK_SIZE],
    int is_start_of_game,
    const int8_t fixed_actions[][DECK_SIZE],
    int8_t mask[ACTION_SPACE_SIZE]
);

#endif // BIG_TWO_CORE_H
